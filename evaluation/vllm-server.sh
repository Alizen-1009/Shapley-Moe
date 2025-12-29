vllm serve \
--model ~/yuhao/hf_models/deepseekv2-lite-coder \
--tensor-parallel-size 8 \
--served-model-name test_model --trust_remote_code --port 8801


