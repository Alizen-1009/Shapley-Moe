vllm serve \
--model ~/yuhao/hf_models/qwen3-30b-a3b \
--tensor-parallel-size 8 \
--served-model-name qwen3-30b-a3b --trust_remote_code --port 8801