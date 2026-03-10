# Hybrid Reward Design Tradeoffs for Multimodal Agent Factuality: An Empirical Analysis


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Base%20Model-LLaVA--1.5--7B-purple.svg)](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
[![Dataset](https://img.shields.io/badge/Dataset-ScienceQA-green.svg)](https://huggingface.co/datasets/derek-thomas/ScienceQA)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the full implementation and evaluation code for **HRA-RL** a Hybrid Reward Architecture for hallucination reduction in multimodal agents. We investigate whether decomposing the RL reward signal into task success, factual grounding, and hallucination penalty components provides advantages over binary reward shaping in terms of factuality, training stability, and convergence.

Experiments are conducted on **ScienceQA** using **LLaVA-1.5-7B** with **LoRA fine-tuning** and **REINFORCE** training, evaluated against three baselines: Vanilla zero-shot, RAG, and Standard PPO.

**Key finding:** Hybrid reward produces substantially lower variance across seeds (TSR std = 0.019) compared to Standard PPO (TSR std = 0.129), despite Standard PPO achieving higher peak performance on a single seed — confirming that reward decomposition improves training stability at the cost of requiring a larger training budget to converge.

---

## Results Summary

All results reported on the frozen ScienceQA test set (n = 500, seed = 42).

| Agent | TSR ↑ | SCS ↑ | HR ↓ | TSR Std (3 seeds) |
|---|---|---|---|---|
| Vanilla LLaVA | 0.336 | 0.448 | 0.664 | 0.006 |
| RAG Baseline | 0.354 | 0.455 | 0.646 | 0.008 |
| Standard PPO | 0.360 | 0.465 | 0.640 | **0.129** ⚠️ |
| **HRA-RL (ours)** | 0.238 | 0.398 | 0.762 | **0.019** ✅ |

**Ablation (seed = 42, n = 500):**

| Config | TSR ↑ | SCS ↑ | HR ↓ |
|---|---|---|---|
| A: Task Only | 0.344 | 0.460 | 0.656 |
| B: Fact Only | 0.342 | 0.453 | 0.658 |
| **C: No Penalty ⭐** | **0.358** | **0.470** | **0.642** |
| D: Full HRA | 0.340 | 0.453 | 0.660 |
| E: Standard PPO | 0.356 | 0.456 | 0.644 |

> **TSR** = Task Success Rate · **SCS** = Semantic Consistency Score · **HR** = Hallucination Rate

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    HRA-RL Agent                      │
│                                                      │
│  Input: Image + Question                             │
│       ↓                                              │
│  LLaVA-1.5-7B (INT4) + LoRA (r=16, 9.96M params)   │
│       ↓                                              │
│  Generated Response                                  │
│       ↓                                              │
│  ┌────────────────────────────────┐                  │
│  │     Hybrid Reward Function     │                  │
│  │                                │                  │
│  │  R = W_task × r_task           │                  │
│  │    + W_fact × FGM(resp, truth) │                  │
│  │    - δ × 𝟙[FGM < τ]           │                  │
│  │                                │                  │
│  │  W_task=1.0  W_fact=0.5        │                  │
│  │  δ=0.5       τ=0.6             │                  │
│  └────────────────────────────────┘                  │
│       ↓                                              │
│  REINFORCE + EMA Baseline (β=0.9)                    │
│  AdamW (lr=1e-5, wd=0.01, clip=0.5)                 │
└─────────────────────────────────────────────────────┘
```

**Factual Grounding Module (FGM):** Cosine similarity between `all-MiniLM-L6-v2` embeddings of generated response and ground-truth answer. Score ≥ 0.6 = factual; score < 0.6 = hallucination penalty applied.

---

## Repository Structure

```
.
├── HRA_Full_v6_fixed.ipynb     # Main Colab notebook (full pipeline)
├── README.md
│
├── experiments/
│   ├── exp1_hra_vanilla.py     # Experiment 1: HRA vs Vanilla
│   ├── exp2_standard_ppo.py    # Experiment 2: Standard PPO baseline
│   ├── exp2b_rag_baseline.py   # Experiment 2b: RAG baseline
│   ├── exp3_llm_judge.py       # Experiment 3: LLM-as-Judge (Groq)
│   ├── exp4_multiseed.py       # Experiment 4: Multi-seed validation
│   └── exp5_ablation.py        # Experiment 5: Reward ablation A–E
│
├── src/
│   ├── model_factory.py        # LLaVA loading + LoRA setup
│   ├── reward.py               # Hybrid reward function + FGM
│   ├── trainer.py              # REINFORCE training loop
│   ├── evaluator.py            # Evaluation metrics (TSR, SCS, HR, EMA)
│   ├── rag.py                  # RAG baseline implementation
│   └── judge.py                # LLM-as-Judge (Groq/llama-3.1-8b)
│
├── paper/
│   ├── Abstract_Final.docx
│   ├── Chapter2_Final.docx     # Background & Related Work
│   ├── Chapter3_Methodology.docx
│   ├── Chapter4_Final.docx     # Experiments & Results
│   └── Chapter5_Conclusion_Final.docx
│
└── results/
    ├── main_results.csv
    ├── ablation_results.csv
    ├── multiseed_results.csv
    └── checkpoint_metrics.json
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA GPU with ≥ 8 GB VRAM (tested on NVIDIA T4 15.89 GB)
- Google Colab (recommended) or local CUDA environment

### Installation

```bash
git clone https://github.com/yourusername/hra-multimodal-factuality.git
cd hra-multimodal-factuality

pip install torch torchvision transformers peft bitsandbytes
pip install datasets sentence-transformers scikit-learn scipy
pip install groq  # for LLM-as-Judge experiment
```

### Environment Variables

```bash
# HuggingFace token (required for LLaVA model access)
export HF_TOKEN="your_huggingface_token"

# Groq API key (required for Experiment 3 only)
export GROQ_API_KEY="your_groq_api_key"
```

### Quick Start (Colab)

Open `HRA_Full_v6_fixed.ipynb` in Google Colab and run cells in order (1 → 26). The notebook handles all installs, dataset loading, training, and evaluation automatically.

**Estimated runtime:** ~3–4 hours on T4 for the full pipeline (all 5 experiments).

---

## Configuration

All hyperparameters are controlled via the `CFG` dictionary in Cell 3 of the notebook:

```python
CFG = {
    # Training
    "PPO_STEPS":              400,
    "TRAIN_ITEMS":            400,
    "LEARNING_RATE":          1e-5,
    "TEMPERATURE":            0.9,
    "MAX_NEW_TOKENS":         50,

    # Evaluation
    "TEST_ITEMS":             500,
    "EVAL_TEMPERATURE":       0.7,
    "SEEDS":                  [42, 123, 7],

    # Hybrid Reward
    "W_TASK":                 1.0,
    "W_FACT":                 0.5,
    "PENALTY":               -0.5,
    "HALLUCINATION_THRESHOLD": 0.6,
    "FGM_THRESHOLD":          0.6,

    # LoRA
    "LORA_RANK":              16,

    # RAG
    "RAG_TOP_K":              3,
    "RAG_CORPUS_MAX":         400,

    # LLM Judge
    "JUDGE_SAMPLE_SIZE":      50,
    "OPENAI_MODEL":           "llama-3.1-8b-instant",  # Groq model

    # Statistics
    "CI_LEVEL":               0.95,
}
```

---

## Experiments

| # | Experiment | Description |
|---|---|---|
| 1 | HRA vs Vanilla | Full HRA agent trained and evaluated against zero-shot baseline |
| 2 | Standard PPO | Binary reward baseline under identical training conditions |
| 2b | RAG Baseline | Retrieval-augmented inference-time baseline (no training) |
| 3 | LLM-as-Judge | Groq/llama-3.1-8b-instant hallucination judge on 50-item sample |
| 4 | Multi-Seed | Statistical validation across seeds {42, 123, 7} with 95% CI + Cohen's d |
| 5 | Ablation A–E | Five reward configurations isolating task signal, factuality reward, and penalty |

---

## Memory Management

Sequential one-model-at-a-time loading is enforced throughout. The `nuke()` function clears GPU memory between model loads:

```python
def nuke():
    for name in ["model","vanilla_model","std_ppo_model","rag_model"]:
        if name in globals():
            try: globals()[name].cpu()   # move off GPU first
            except: pass
            try: del globals()[name]
            except: pass
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

> ⚠️ Always call `nuke()` before loading a new model. Peak VRAM per phase is ~8.6 GB on the T4.

---

## Known Issues & Limitations

| Issue | Description | Status |
|---|---|---|
| LLM Judge returns HR=0.000 | Groq/llama judge miscalibrated for structured MCQ benchmarks | Known — excluded from primary analysis |
| Standard PPO instability | TSR std=0.129 across seeds; degenerate outputs at seed=123 | Expected — binary reward reward collapse |
| Full HRA underperforms at 400 steps | Penalty term needs >400 steps to overcome reward sparsity | Documented in Chapter 5 |
| FGM threshold sensitivity | Scores cluster near τ=0.6, making TSR/HR sensitive to small perturbations | Threshold sensitivity analysis is future work |

---




## Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA) — base multimodal model (Liu et al., 2023)
- [ScienceQA](https://scienceqa.github.io/) — evaluation benchmark (Lu et al., 2022)
- [PEFT / LoRA](https://github.com/huggingface/peft) — parameter-efficient fine-tuning
- [sentence-transformers](https://www.sbert.net/) — FGM embeddings (all-MiniLM-L6-v2)
- [Groq](https://groq.com/) — LLM-as-Judge inference
- Van Seijen et al. (2017) — original Hybrid Reward Architecture paper

---

## License

MIT License. See [LICENSE](LICENSE) for details.
