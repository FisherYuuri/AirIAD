# AirIAD: Agentic Iterative Reasoning for Industrial Anomaly Detection

## Table of Contents

- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Stage 1: SFT Supervised Fine-Tuning](#stage-1-sft-supervised-fine-tuning)
- [Stage 2: SSGiGPO Reinforcement Learning](#stage-2-ssgigpo-reinforcement-learning)
- [Core Components](#core-components)

## Project Structure

```
opensource/
├── dataset/                          # Training/test data
│   ├── train-sft1600.json            # SFT training set (1600 samples, ShareGPT multi-turn format)
│   ├── train-grpo366.json            # GRPO training set (366 samples, with gt/bbox/correct_answer)
│   └── test6400.json                 # Test set (6400 samples)
├── dataset_prepare/                  # Data preparation scripts
│   ├── prepare_sft_data.py           # SFT data: crop images and update JSON paths
│   ├── image_compression.py          # Image compression: resize MMAD images (max side ≤ 1024, keep aspect ratio)
│   └── convert_to_verl_parquet.py    # Format conversion: JSON → verl parquet format
├── eval/                             # Evaluation scripts
│   ├── eval_agent_vllm.py            # vLLM multi-turn tool-call inference evaluation
│   ├── eval_results_metrics.py       # Metric computation
│   └── analyze_multiturn_effect.py   # Multi-turn effect analysis (self-correction statistics)
├── results/                          # Pre-computed results
│   ├── outputs_airiad-4b-os.jsonl    # 4B one-shot inference outputs
│   ├── outputs_airiad-4b-zs.jsonl    # 4B zero-shot inference outputs
│   ├── outputs_airiad-8b-os.jsonl    # 8B one-shot inference outputs
│   ├── outputs_airiad-8b-zs.jsonl    # 8B zero-shot inference outputs
│   ├── results_airiad-4b-os.jsonl    # 4B one-shot metric results
│   ├── results_airiad-4b-zs.jsonl    # 4B zero-shot metric results
│   ├── results_airiad-8b-os.jsonl    # 8B one-shot metric results
│   └── results_airiad-8b-zs.jsonl    # 8B zero-shot metric results
├── knowledge_base/                   # Domain knowledge base
│   ├── knowledge.json                # Defect descriptions per dataset/category
│   └── candidates.json               # Candidate type mapping (variant → canonical exact match)
├── LlamaFactory-main/                # LlamaFactory (SFT stage)
│   └── LlamaFactory-main/
│       ├── data/dataset_info.json    # Dataset registry (includes airiad entry)
│       └── examples/                 # Training config templates
├── verl-main/                        # verl (SSGiGPO stage)
│   ├── verl/
│   │   ├── tools/airiad_tool.py      # inspect(crop, query) tool implementation
│   │   ├── utils/reward_score/
│   │   │   ├── airiad_reward.py      # AirIAD reward function (outcome + step rewards)
│   │   │   └── __init__.py           # Registers "agentiad/grpo" data source
│   │   ├── trainer/ppo/core_ssgigpo.py # SSGiGPO algorithm implementation
│   │   └── experimental/agent_loop/
│   │       └── airiad_agent_loop.py  # AirIAD multi-turn Agent Loop
│   └── examples/sglang_multiturn/
│       ├── config/
│       │   ├── airiad_multiturn_ssgigpo_4b.yaml   # 4B SSGiGPO config
│       │   ├── airiad_multiturn_ssgigpo_8b.yaml   # 8B SSGiGPO config
│       │   └── tool_config/airiad_tool_config.yaml  # Tool config
│       ├── run_qwen3_vl-4b_airiad_multiturn_ssgigpo.sh  # 4B training script
│       └── run_qwen3_vl-8b_airiad_multiturn_ssgigpo.sh  # 8B training script
```

## Evaluation

### Quick Start: Evaluate Pre-computed Results

We provide pre-computed inference outputs and metric results in the `results/` directory. You can directly compute metrics from these files without running model inference:

```bash
cd eval

# Compute metrics from pre-computed results
python eval_results_metrics.py --input ../results/results_airiad-4b-zs.jsonl
python eval_results_metrics.py --input ../results/results_airiad-4b-os.jsonl
python eval_results_metrics.py --input ../results/results_airiad-8b-zs.jsonl
python eval_results_metrics.py --input ../results/results_airiad-8b-os.jsonl
```

| File | Model | Mode | Description |
|------|-------|------|-------------|
| `results_airiad-4b-zs.jsonl` | 4B | Zero-shot | 4B model, target image only |
| `results_airiad-4b-os.jsonl` | 4B | One-shot | 4B model, with reference normal image |
| `results_airiad-8b-zs.jsonl` | 8B | Zero-shot | 8B model, target image only |
| `results_airiad-8b-os.jsonl` | 8B | One-shot | 8B model, with reference normal image |
| `outputs_airiad-4b-zs.jsonl` | 4B | Zero-shot | Raw inference outputs |
| `outputs_airiad-4b-os.jsonl` | 4B | One-shot | Raw inference outputs |
| `outputs_airiad-8b-zs.jsonl` | 8B | Zero-shot | Raw inference outputs |
| `outputs_airiad-8b-os.jsonl` | 8B | One-shot | Raw inference outputs |

### Evaluation with vLLM

To run inference on your own model:

```bash
cd eval

# Zero-shot mode
python eval_agent_vllm.py \
    --model /path/to/model \
    --input ../dataset/test6400.json \
    --output results.jsonl \
    --knowledge ../knowledge_base/knowledge.json \
    --candidates ../knowledge_base/candidates.json

# One-shot mode (with reference normal image)
python eval_agent_vllm.py \
    --oneshot \
    --model /path/to/model \
    --input ../dataset/test6400.json \
    --output results_oneshot.jsonl \
    --knowledge ../knowledge_base/knowledge.json \
    --candidates ../knowledge_base/candidates.json
```

### Compute Evaluation Metrics

```bash
cd eval

python eval_results_metrics.py --input ../results/results_airiad-4b-zs.jsonl
```

Evaluation metrics include:
- **Detection Accuracy**: Whether the model correctly identifies normal/anomalous
- **Anomaly Type Accuracy**: Defect type classification accuracy
- **IoU**: Intersection over Union of localization bounding boxes
- **Joint Accuracy**: Joint accuracy of detection + type + localization

### Analyze Multi-Turn Effect

Analyze the self-correction capability of multi-turn reasoning (requires `outputs_*.jsonl` files with message-level data):

```bash
cd eval

python analyze_multiturn_effect.py --input ../results/outputs_airiad-4b-zs.jsonl
python analyze_multiturn_effect.py --input ../results/outputs_airiad-8b-os.jsonl
```

Four statistics are reported:
- **Cat 1 — Normal queried then output normal**: Among normal samples that called inspect(), the proportion that ultimately output "normal"
- **Cat 2 — Type correction**: Proportion of multi-turn anomaly samples where an intermediate query was wrong but the final type was correct
- **Cat 3 — IoU improvement**: Proportion of multi-turn anomaly samples where the final answer IoU exceeds the intermediate crop IoU

## Environment Setup

### Install Dependencies

#### 1. LlamaFactory (SFT Stage)

Refer to the [LlamaFactory installation guide](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md) for detailed setup instructions.

```bash
cd LlamaFactory-main/LlamaFactory-main
pip install -e ".[torch,metrics]"
```

#### 2. verl (SSGiGPO Stage)

Use the official verl Docker image:

```bash
docker pull verlai/verl:sgl059.dev3
docker start verl
docker exec -it verl bash

# Inside the container
git clone https://github.com/volcengine/verl && cd verl
pip3 install --no-deps -e .
pip3 install -e .[vllm]      # Install vLLM
pip3 install -e .[sglang]    # Install SGLang
```

#### 3. Additional Dependencies

```bash
pip install qwen-vl-utils
```

## Data Preparation

### 1. Download MMAD Dataset

Download the MMAD dataset from HuggingFace:

```bash
# Download from https://huggingface.co/datasets/jiang-cc/MMAD
# Assume saved to /path/to/MMAD
```

### 2. Image Compression

Resize original images so that the maximum side is at most 1024 pixels (keep aspect ratio):

```bash
cd dataset_prepare

# Modify paths in image_compression.py, then run
python image_compression.py
```

### 3. SFT Data Preparation

Crop inspect images from `train-sft1600.json` and update paths:

```bash
python prepare_sft_data.py \
    --input dataset/train-sft1600.json \
    --mmap_root /path/to/MMAD \
    --output dataset/train-sft1600-ready.json \
    --crop_dir dataset/sft_crops
```

Copy the prepared SFT data to LlamaFactory's data directory:

```bash
cp dataset/train-sft1600-ready.json LlamaFactory-main/LlamaFactory-main/data/train-sft1600.json
```

### 4. GRPO Data Preparation (verl parquet format)

Convert GRPO training data to the parquet format required by the verl framework:

```bash
python convert_to_verl_parquet.py \
    --train_file dataset/train-grpo366.json \
    --test_file dataset/test6400.json \
    --output_dir /path/to/verl/data \
    --data_source agentiad/grpo \
    --mode mix
# Output: /path/to/verl/data/train_mix.parquet, test_os.parquet
```

> **Note**: `--mode mix` alternates between zero-shot (target image only) and one-shot (with a reference normal image) modes.

## Stage 1: SFT Supervised Fine-Tuning

### Register Dataset

Ensure the `airiad` dataset is registered in `LlamaFactory-main/LlamaFactory-main/data/dataset_info.json`:

```json
{
  "airiad": {
    "file_name": "train-sft1600.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

### LoRA Fine-Tuning

```bash
cd LlamaFactory-main/LlamaFactory-main

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/qwen3vl_lora_sft.yaml

bash examples/extras/fsdp_qlora/train.sh
```

### Merge LoRA Weights

After SFT training, merge LoRA weights for use in the SSGiGPO stage:

```bash
llamafactory-cli export examples/merge_lora/qwen3vl_lora_sft_merge.yaml
```

## Stage 2: SSGiGPO Reinforcement Learning

### Training (4B Model)

```bash
cd verl-main

# Set SFT model path
export MODEL_PATH=/path/to/model

bash examples/sglang_multiturn/run_qwen3_vl-4b_airiad_multiturn_ssgigpo.sh
```

### Training (8B Model)

```bash
export MODEL_PATH=/path/to/model

bash examples/sglang_multiturn/run_qwen3_vl-8b_airiad_multiturn_ssgigpo.sh
```

Key parameter differences between 4B and 8B:

| Parameter | 4B | 8B |
|-----------|-----|-----|
| `data.train_batch_size` | 64 | 32 |
| `data.max_prompt_length` | 3072 | 3072 |
| `data.max_response_length` | 8096 | 8096 |
| `trainer.total_training_steps` | 80 | 160 |

## Core Components

### inspect(crop, query) Tool

The model's sole tool, defined in `verl-main/verl/tools/airiad_tool.py`:

- **crop**: `[x1, y1, x2, y2]`, coordinate range [0, 1000], representing the image region to crop
- **query**: An ABCD letter, selecting the anomaly type from the candidate list that best matches the current hypothesis
- **Returns**: The cropped image + domain knowledge retrieved via KnowledgeIndex exact matching + IoU step-level reward

### Reward Function

Defined in `verl-main/verl/utils/reward_score/airiad_reward.py`:

**Outcome Reward** (R_outcome, counts toward total score):
- has_anomaly correct: +1.0
- anomaly_type correct: +1.0
- IoU reward (dynamic threshold): +≤1.0
- Joint alignment bonus: +IoU × type_match

**Step Rewards** (used for SSGiGPO step-level advantage, not counted in total score):
- Category correct: +1
- IoU ≥ dynamic threshold: +1
- Joint alignment bonus: +IoU × type_match

### Domain Knowledge Base

- `knowledge_base/knowledge.json`: Defect descriptions for each category across datasets
- `knowledge_base/candidates.json`: Variant → canonical exact match mapping, used for KnowledgeIndex retrieval
