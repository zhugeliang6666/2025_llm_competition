#!/bin/bash
set -e

# qwen_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/zy_model_path/Qwen2-72B-Instruct
# llama_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/llama3-70B
# gemma_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it



qwen_path=/data/oceanus_ctr/j-hewuqing-jk/models/Qwen/Qwen2___5-72B-Instruct/
gemma_path=/data/oceanus_ctr/j-hewuqing-jk/models/LLM-Research/gemma-2-9b-it/


fold=$1
echo run:${fold}

# train qwen2 70b
sh run_fintune.sh qwen2 ${qwen_path} ${fold}
# # predict train logits
python predict_train.py ${qwen_path} ../model_save/qwen2_4bit_load_fintune/epoch_0_model/adapter.bin ../data/processed_data/qwen2fold${fold}/train.parquet ../data/oof/qwen2fold${fold}_train.parquet

# # merge  logits 
python merge_logits.py ../data/processed_data/gemma2fold${fold}/train.parquet ../data/oof/qwen2fold${fold}_train.parquet ../data/processed_data/gemma2fold${fold}/train_logits.parquet

# distill fintune gemma2-9b
sh run_fintune_16bit_distill.sh gemma2 ${gemma_path} ${fold}
