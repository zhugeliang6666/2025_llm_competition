import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pickle
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import sys
import gc
model_path = sys.argv[1]
save_name = sys.argv[2]
print("model_path:", model_path)
print("save_name:", save_name)

### load tokenizer
MODEL_NAME = model_path
MAX_LENGTH = 3072
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = tokenizer.eos_token_id

# load data
train0 = pd.read_parquet("../data/train.parquet")
# train['prompt'] = train['prompt'].apply(lambda x: "\n".join(eval(x)))
train = pd.read_parquet('../data/ultrafeedback_lmsysformat.parquet')

train = train[train['winner_tie']==0]

train = train.drop_duplicates(subset='prompt')
train = train.sample(frac=0.3, random_state=42)

sample_df = train.sample(frac=0.5, random_state=42)
sample_df['response_a'], sample_df['response_b'] = sample_df['response_b'], sample_df['response_a']
sample_df['winner_model_a'], sample_df['winner_model_b'] = sample_df['winner_model_b'], sample_df['winner_model_a']
train.update(sample_df)

train['prompt'] = train['prompt'].apply(lambda x: str([x]))

train['prompt'] = train['prompt'].apply(lambda x: "\n".join(eval(x)))

# 新建winner列
def determine_winner(row):
    if row['winner_model_a'] == 1:
        return 'model_a'
    else:
        return 'model_b'

train['winner'] = train.apply(determine_winner, axis=1)
del train['winner_tie']
del train['model_a']
del train['model_b']
del train['winner_model_a']
del train['winner_model_b']
del train0['language']
gc.collect()

train = pd.concat([train0, train], axis=0, ignore_index=True)


def do(row):
    # assert row['winner_tie'] + row['winner_model_a'] + row['winner_model_b'] == 1
    if row['winner'] == 'model_a':
        return "A"
    else:
        return "B"


train['label'] = train.apply(lambda row: do(row), axis=1)
print(train['label'].value_counts())


def do(x):
    try:
        return "\n".join(eval(x))
    except:
        x = x.replace("[", "").replace("]", "").strip()
        return x


train['response_a_str'] = train['response_a'].apply(lambda x: do(x))
train['response_b_str'] = train['response_b'].apply(lambda x: do(x))

######################### split 5 fold ###############################
# train = train.reset_index(drop=True)
# train['fold'] = 0
# skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
# for fold, (train_index, test_index) in enumerate(skf.split(train, train['label'].values)):
#     train.loc[test_index, 'fold'] = fold
# print(train['fold'].value_counts())


######################### get the input prompt ###############################
def create_rounds(query, answer_a, answer_b):
    prompt = f"""User question:
\"""{query}\"""
Answer A:
\"""{answer_a}\"""
Answer B:
\"""{answer_b}\"""
"""
    return prompt


texts = []
texts_token_len = []
for _, row in tqdm(train.iterrows()):
    query = ' '.join(row['prompt'].split(' ')[:1024])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:2048])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:2048])
    prompt_len = 1024
    query_len = len(tokenizer.encode(query))
    answer_a_len = len(tokenizer.encode(answer_a))
    answer_b_len = len(tokenizer.encode(answer_b))
    if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
        query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
            tokenizer.encode(query)[:prompt_len])
        query_len = len(tokenizer.encode(query))
        if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
            remain_len = MAX_LENGTH - query_len
            token_answer_a = tokenizer.encode(answer_a)
            token_answer_b = tokenizer.encode(answer_b)
            if len(token_answer_a) > len(token_answer_b):
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
                        token_answer_b):
                    token_answer_a = token_answer_a[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            else:
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
                        token_answer_a):
                    token_answer_b = token_answer_b[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            answer_a = tokenizer.decode(token_answer_a)
            answer_b = tokenizer.decode(token_answer_b)
    prompt = create_rounds(query, answer_a, answer_b)
    texts.append(prompt)
train['text'] = texts[:]

train_reverse = train.copy()
def do(row):
    # assert row['winner_tie'] + row['winner_model_a'] + row['winner_model_b'] == 1
    if row['winner'] == 'model_a':
        return "B"
    else:
        return "A"
train_reverse['label'] = train_reverse.apply(lambda row: do(row), axis=1)
texts = []
texts_token_len = []
for _, row in tqdm(train_reverse.iterrows()):
    query = ' '.join(row['prompt'].split(' ')[:1024])
    answer_a = ' '.join(row['response_a_str'].split(' ')[:2048])
    answer_b = ' '.join(row['response_b_str'].split(' ')[:2048])
    prompt_len = 1024
    query_len = len(tokenizer.encode(query))
    answer_a_len = len(tokenizer.encode(answer_a))
    answer_b_len = len(tokenizer.encode(answer_b))
    if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
        query = query if len(tokenizer.encode(query)) < prompt_len else tokenizer.decode(
            tokenizer.encode(query)[:prompt_len])
        query_len = len(tokenizer.encode(query))
        if query_len + answer_a_len + answer_b_len > MAX_LENGTH:
            remain_len = MAX_LENGTH - query_len
            token_answer_a = tokenizer.encode(answer_a)
            token_answer_b = tokenizer.encode(answer_b)
            if len(token_answer_a) > len(token_answer_b):
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_a) > len(
                        token_answer_b):
                    token_answer_a = token_answer_a[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            else:
                while len(token_answer_a) + len(token_answer_b) > remain_len and len(token_answer_b) > len(
                        token_answer_a):
                    token_answer_b = token_answer_b[:-1]
                while len(token_answer_a) + len(token_answer_b) > remain_len:
                    token_answer_a = token_answer_a[:-1]
                    if len(token_answer_a) + len(token_answer_b) > remain_len:
                        token_answer_b = token_answer_b[:-1]
            answer_a = tokenizer.decode(token_answer_a)
            answer_b = tokenizer.decode(token_answer_b)
    prompt = create_rounds(query, answer_b, answer_a)
    texts.append(prompt)
train_reverse['text'] = texts[:]
train_reverse['reverse'] = True

train['reverse'] = False
train = pd.concat([train, train_reverse], axis=0)
train['order_index'] = list(range(len(train)))


# ######################### save the data ###############################
def do(x):
    if x == "C":
        return 2
    elif x == "B":
        return 1
    else:
        return 0


train['label'] = train['label'].apply(lambda x: do(x))
print(train.shape)
with open(f"../data/processed_data/{save_name}fold5/train.parquet", 'wb') as f:
    pickle.dump(train, f)
