# AirIAD: Agentic Iterative Reasoning for Industrial Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

<p align="center">
  <img src="assets/airiadoverview.png" alt="AirIAD Framework Overview" width="100%">
</p>

## 📖 Abstract


## 🔗 Model Weights (ModelScope)
We provide our fine-tuned model weights openly via ModelScope. AirIAD is built upon the powerful **Qwen3-VL-Instruct** architecture, offering two scalable versions to balance deployment efficiency and reasoning capability.

| Model Version | Base Model | Parameters | ModelScope Link |
| :--- | :--- | :--- | :--- |
| **AirIAD-4B** | Qwen3-VL-Instruct-4B | 4B | [🔗 FisherYuuri/AirIAD-4B](https://modelscope.cn/models/FisherYuuri/AirIAD-4B) |
| **AirIAD-8B** | Qwen3-VL-Instruct-8B | 8B | [🔗 FisherYuuri/AirIAD-8B](https://modelscope.cn/models/FisherYuuri/AirIAD-8B) |

## 📅 TODO
- [ ] Release Training Scripts (Support for both Qwen3-VL-Instruct 4B and 8B versions).
- [ ] Release Testing & Evaluation Scripts for the MMAD benchmark.
- [ ] Open-source the training datasets (Warm-up SFT & RL data).

## 🙏 Acknowledgments
We would like to express our sincere gratitude to the following outstanding open-source projects and pioneering research works, which greatly inspired and facilitated the development of AirIAD:

* **Training Frameworks:** We thank the developers of [verl](https://github.com/volcengine/verl) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for their robust and efficient training infrastructures.
* **Benchmarks & Datasets:** Special thanks to the creators of [MMAD](https://github.com/jam-cc/MMAD) for providing comprehensive benchmarks that drive IAD research forward.
* **Related MLLM & Reasoning Frameworks:** We draw inspiration from excellent repositories including [M3-AD](https://github.com/Yanhui-Lee/M3-AD), [JUDO](https://github.com/woodavid31/JUDO), [IAD-R1](https://github.com/Yanhui-Lee/IAD-R1), [AgentIAD](https://arxiv.org/abs/2512.13671) and [AnomalyR1](https://github.com/chaoyuhao/AnomalyR1).
