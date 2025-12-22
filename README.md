# Shapley-Moe

## 1.Descripts
Shapley Value guide Moe prune

## 2.Env List
lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/main

```pip install "lm_eval[vllm]"```

## 3.Steps
### 3.0 get baseline
using ```run_lm-eval-vllm-dp8.sh``` and defualt model_path to get baseline

### 3.1 dataset preparation
prepare necessary dataset(e.g gsm8k), defualt 25 data
using ```/dataset/download.sh```


### 3.2 few-shot 
using dataset to find differnet layers expert activate set
using ```/few-shot/run_analysis.sh``` and ```3.1 dataset```


### 3.3 calc shapley value
using exprt activate set to calcuate every expert's shapley value per layer
using ```/calc_shapley/calc_shaple.sh``` and ```3.2 results```

### 3.4 expert select 
select expert form it's shapley value
using ```/expert_select/run_selection_by_rate.sh``` and ```3.3 results``` to find needed expert by global expert keeprate, can choose per layer
using ```/expert_select/run_selection.sh``` and ```3.3 results``` to find needed expert by shapley keeprate

### 3.5 model_save
save pruned model to safetensor, directly mask unselected experts wights to zero
using ```/model_save/run_sace_model.sh``` and ```3.4 results``` to save model

### 3.6 test pruned model acc
using ```run_lm-eval-vllm-dp8.sh``` and prund model_path to get pruned score

