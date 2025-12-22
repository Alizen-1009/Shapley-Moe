#!/bin/bash

export HF_ALLOW_CODE_EVAL="1"

lm_eval \
  --model vllm \
  --model_args pretrained=/root/yuhao/hf_models/qwen3-30b-a3b,dtype=auto,tensor_parallel_size=8,trust_remote_code=True \
  --tasks mmmu_val_finance \
  --batch_size auto \
  --confirm_run_unsafe_code \
  --num_fewshot 5 \
  --apply_chat_template \
  
  

# lm_eval \
#   --model vllm \
#   --model_args pretrained=/root/yuhao/hf_models/gpt-oss-20b,dtype=auto,data_parallel_size=8,gpu_memory_utilization=0.9,max_model_len=4096,trust_remote_code=True  \
#   --tasks mmlu_elementary_mathematics \
#   --confirm_run_unsafe_code \
#   --apply_chat_template \
#   --batch_size auto \


# lm_eval --model hf \
#     --tasks gsm8k \
#     --model_args pretrained=/root/yuhao/hf_models/OLMOE-7B-Instruct,dtype=auto,parallelize=True \
#     --batch_size auto


# agieval_gaokao_mathqa
