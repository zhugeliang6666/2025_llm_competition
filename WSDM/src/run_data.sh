#!/bin/bash


# qwen_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/zhouyang96/zy_model_path/Qwen2-72B-Instruct
# llama_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/llama3-70B
# gemma_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/gemma-2-9b-it


qwen_path=/data/oceanus_ctr/j-hewuqing-jk/models/Qwen/Qwen2___5-72B-Instruct/
gemma_path=/data/oceanus_ctr/j-hewuqing-jk/models/LLM-Research/gemma-2-9b-it/


python prepare_data_ut.py ${qwen_path} qwen2
python prepare_data_ut.py ${gemma_path} gemma2


