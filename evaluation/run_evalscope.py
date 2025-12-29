from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="test_model",  # 模型名称
    api_url="http://127.0.0.1:8801/v1",  # 模型服务地址
    eval_type=EvalType.SERVICE,  # 评测类型，这里使用服务评测
    # datasets=["arc"],  # 评测数据集列表
    datasets=["humaneval", "gsm8k", "aime24", "med_mcqa", "arc",
              "gpqa_diamond", "logi_qa", "truthful_qa", "ontonotes5", "math_500", "biomix_qa", 
              "aime25", "pubmedqa", "live_code_bench"], 
    # generation_config={
    #     "extra_body": {"reasoning_effort": "high"}  # 模型生成参数，这里设置为高推理水平
    # },
    generation_config={
        'max_tokens': 20480,  # 设置最大生成长度
        # 'extra_body':{'chat_template_kwargs': {'enable_thinking': True},} 
    },
    eval_batch_size=128,  # 并发测试的batch size
    timeout=60000,  # 超时时间，单位为秒
    
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
