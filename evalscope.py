from evalscope.constants import EvalType
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="qwen3-30b-a3b",  # 模型名称
    api_url="http://127.0.0.1:8801/v1",  # 模型服务地址
    eval_type=EvalType.SERVICE,  # 评测类型，这里使用服务评测
    datasets=["aime24"],  # 测试的数据集
    # generation_config={
    #     "extra_body": {"reasoning_effort": "high"}  # 模型生成参数，这里设置为高推理水平
    # },
    eval_batch_size=30,  # 并发测试的batch size
    timeout=60000,  # 超时时间，单位为秒
)

run_task(task_cfg=task_cfg)


# evalscope eval \
#  --model OLMOE-7B-Instruct \
#  --api-url http://127.0.0.1:8801/v1 \
#  --eval-type openai_api \
#  --datasets aime24
