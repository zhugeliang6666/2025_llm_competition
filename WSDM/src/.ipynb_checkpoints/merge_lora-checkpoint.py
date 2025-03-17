import time
from dataclasses import dataclass
import pickle
import torch
import sklearn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import os
adapter_pathz = [
    # '../model_save/gemma2fold0_16bit_load_fintune/best_val_loss_model/adapter.bin',
    # '../model_save/gemma2fold1_16bit_load_fintune/best_val_loss_model/adapter.bin',
    # '../model_save/gemma2fold2_16bit_load_fintune/best_val_loss_model/adapter.bin',
    # '../model_save/gemma2fold3_16bit_load_fintune/best_val_loss_model/adapter.bin',
    '../model_save/gemma2fold4_16bit_load_fintune/best_val_loss_model/adapter.bin'
]
loaded_adapters = []
for path in adapter_pathz:
    if os.path.exists(path):
        print(f'Loading {path}')
        state_dict = torch.load(path)
        loaded_adapters.append(state_dict)
    else:
        print(f'Path not found, skipping: {path}')

if not loaded_adapters:
    raise ValueError('No adapters were loaded,pls check!')

n=len(loaded_adapters)
print(f'We need to combine {n} adapters files.')

combined_dict={}

first_key = loaded_adapters[0].keys()
for k in first_key:
    summed= None
    for state_dict in loaded_adapters:
        if summed is None:
            summed = state_dict[k]
        else:
            summed += state_dict[k]
    combined_dict[k] = summed / n

output_path = '../model_save/final_adapter.bin'
torch.save(combined_dict, output_path)
print(f'Final adapter has been saved to: {output_path}')