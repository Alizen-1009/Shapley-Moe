from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="test_model",  # model name
    api_url="http://127.0.0.1:8801/v1",  # model service URL
    eval_type=EvalType.SERVICE,  # evaluation type, using service evaluation here
    # datasets=["arc"],  # evaluation dataset list
    datasets=["humaneval", "gsm8k", "aime24", "med_mcqa", "arc",
              "gpqa_diamond", "logi_qa", "truthful_qa", "ontonotes5", "math_500", "biomix_qa", 
              "aime25", "pubmedqa", "live_code_bench"], 
    # generation_config={
    #     "extra_body": {"reasoning_effort": "high"}  # model generation parameters, set to high reasoning level here
    # },
    generation_config={
        'max_tokens': 20480,  # set max generation length
        # 'extra_body':{'chat_template_kwargs': {'enable_thinking': True},} 
    },
    eval_batch_size=128,  # concurrent test batch size
    timeout=60000,  # timeout in seconds
    
)

run_task(task_cfg=task_cfg)


# evalscope eval \
#  --model OLMOE-7B-Instruct \
#  --api-url http://127.0.0.1:8801/v1 \
#  --eval-type openai_api \
#  --datasets aime24


# gsm8k
# aime24
# humaneval
# live_codebench
# HealthBench
# med_mcqa
# gpqa-diamond
# logi_qa
# truthful_qa
# pubmedqa
# ontonotes5
# math_500
# biomix_qa
# simple_qa
# aime25
