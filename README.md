# nanoGPT 2.0 (nanochat)

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

> **nanoGPT 2.0** is an enhanced, full-stack evolution of Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) — the simplest experimental harness for training LLMs from scratch. This repository pairs nanochat's minimal, hackable PyTorch engine with a **modern streaming Web Frontend UI**, **1-click Kaggle Cloud GPU training integration**, and **production-ready developer tooling**.
>
> 🔗 **Repository**: [https://github.com/NB7551498/nanoGPT-2.0](https://github.com/NB7551498/nanoGPT-2.0)

---

## ⚡ Key Enhancements in nanoGPT 2.0

### 1. 🌐 Interactive Web Frontend (`web_app.py`)
A fast, modern web interface built on **FastAPI**, **Tailwind CSS**, and **HTML5/JS** designed for interactive prompt testing and real-time generation:
- **Real-Time Token Streaming**: Watch responses stream token-by-token directly in the browser.
- **Hyperparameter Tuning Controls**: Live sliders for `Temperature`, `Top-K`, and `Max Tokens`.
- **System Prompt Support**: Customize agent behavior with on-the-fly system prompt instructions.
- **Model Checkpoint Switcher**: Seamlessly switch between `SFT` (Chat), `RL` (Reinforcement Learning), and `Base` checkpoints.
- **Cloud GPU Status Monitor**: Real-time polling badge reflecting the live training status of your remote Kaggle GPU workers.

To launch the web app:
```bash
python web_app.py
# or
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

### 2. 🚀 Kaggle Cloud GPU Training Integration
Train your models for free on Kaggle's cloud GPUs (NVIDIA T4 / P100 / A100):
- **[`nanochat_kaggle_gpu.ipynb`](nanochat_kaggle_gpu.ipynb)**: An end-to-end, self-contained notebook covering:
  1. GPU environment checks and PyTorch CUDA verification.
  2. Dataset shard download and fast BPE tokenizer training.
  3. Base model pretraining with configurable `--depth` and `--device-batch-size`.
  4. Validation evaluation (CORE metric and BPB loss).
  5. Supervised Fine-Tuning (SFT) for instruction following.
  6. Interactive inference testing.
- **Headless Cloud Execution via Kaggle API (`kaggle_kernel/`)**:
  Push and trigger training jobs remotely using Kaggle's CLI:
  ```bash
  python -m kaggle kernels push -p kaggle_kernel
  python -m kaggle kernels status nikhilbhardwaj2003/nanochat-gpu-training
  ```

---

### 3. 📦 Sample Data Included (`data/`)
Ready-to-use sample text corpora (`data/input.txt`, `data/more.txt`) allow you to test tokenizers, run micro-benchmarks, and verify training pipelines immediately without downloading gigabytes of web shards.

---

### 4. 🔒 Credential Security & Git Hardening
Configured with strict `.gitignore` rules to safeguard private tokens, API keys (`kaggle.json`, `.env`), and temporary cache directories.

---

## 📖 About nanochat Core

nanochat is designed to run on a single GPU node. The code is minimal and hackable, covering all major LLM stages: tokenization, pretraining, finetuning, evaluation, and inference. For example, you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $48 (~2 hours of an 8XH100 GPU node) and then talk to it over a simple CLI or the web UI. On a spot instance, total cost can be closer to ~$15.

nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability is approximately depth 26). All other hyperparameters (width, number of heads, learning rates, training horizons, weight decay) are calculated automatically in an optimal way.

---

## 🏆 Time-to-GPT-2 Leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most compute. Inspired by `modded-nanogpt`, nanochat maintains a leaderboard for a "GPT-2 speedrun" (wall-clock time required to reach GPT-2 capability, measured by the DCLM CORE score). The script [runs/speedrun.sh](runs/speedrun.sh) reflects the reference training pipeline.

| # | Time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more details.

---

## 🚀 Getting Started

### 1. Installation

nanochat uses [uv](https://docs.astral.sh/uv/) for fast dependency management:

```bash
# Clone the repository
git clone https://github.com/NB7551498/nanoGPT-2.0.git
cd nanoGPT-2.0

# Install dependencies with uv
uv sync --extra gpu    # For CUDA GPU nodes (A100, H100, etc.)
uv sync --extra cpu    # For CPU-only or Apple Silicon / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):
```bash
uv sync --extra gpu --group dev
```

Alternatively, standard pip installation:
```bash
pip install -e .
pip install fastapi uvicorn
```

---

### 2. Reproduce & Talk to GPT-2 (CLI)

The complete pipeline to train and chat is contained in [runs/speedrun.sh](runs/speedrun.sh):

```bash
bash runs/speedrun.sh
```

Once trained, chat interactively over the CLI:
```bash
python -m scripts.chat_cli
```
Or test a single prompt:
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"
```

---

### 3. Launch the Web Interface

Start the local web server:
```bash
python web_app.py
```
Navigate to **`http://localhost:8000`** to chat, tweak generation parameters, and inspect GPU status.

---

## 🔬 Research & Scaling Laws

For quick experimentation (~5 minute pretraining runs), train a 12-layer model (GPT-1 sized):

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

Monitor your training metrics:
1. `val_bpb`: Validation loss in bits-per-byte (vocab-size invariant).
2. `core_metric`: DCLM CORE score.
3. VRAM utilization, `train/mfu` (Model FLOPS utilization), and throughput (`train/tok_per_sec`).

---

## 💻 Precision & Data Types

nanochat does not use `torch.amp.autocast`. Precision is managed explicitly via `COMPUTE_DTYPE` (defined in `nanochat/common.py`):

| Hardware | Default dtype | Notes |
|----------|---------------|-------|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default; macOS MPS also runs `NANOCHAT_DTYPE=bfloat16` |

Override via environment variable:
```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train
```

---

## 📁 Repository Structure

```
.
├── README.md                       # Documentation & Project Guide
├── web_app.py                      # FastAPI + Tailwind Web UI server
├── nanochat_kaggle_gpu.ipynb       # 1-Click Kaggle GPU Training Notebook
├── kaggle_kernel/                  # Headless Kaggle CLI Kernel deployment
│   ├── kernel-metadata.json
│   └── nanochat_kaggle_gpu.ipynb
├── data/                           # Sample datasets (input.txt, more.txt)
├── dev/                            # Benchmarking & Scaling analyses
│   ├── nanochat.png
│   └── repackage_data_reference.py
├── nanochat/                       # Core model library
│   ├── checkpoint_manager.py       # Checkpoint save/load
│   ├── common.py                   # Compute init, logging, dtype config
│   ├── core_eval.py                # DCLM CORE benchmark
│   ├── dataloader.py               # Tokenizing distributed dataloader
│   ├── dataset.py                  # Dataset shard downloading utils
│   ├── engine.py                   # Inference engine with KV cache
│   ├── execution.py                # Sandboxed Python code tool execution
│   ├── flash_attention.py          # Attention kernel wrapper
│   ├── fp8.py                      # FP8 primitives
│   ├── gpt.py                      # GPT Transformer nn.Module architecture
│   ├── loss_eval.py                # Bits-per-byte loss evaluation
│   ├── optim.py                    # Muon + AdamW distributed optimizers
│   └── tokenizer.py                # BPE Tokenizer (GPT-4 style)
├── runs/                           # Ready-to-run shell recipes
│   ├── miniseries.sh               # Miniseries training sweeps
│   ├── runcpu.sh                   # Small CPU/MPS training run
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Reference GPT-2 speedrun recipe
├── scripts/                        # Standalone execution entrypoints
│   ├── base_eval.py                # Base model evaluation
│   ├── base_train.py               # Pretraining script
│   ├── chat_cli.py                 # Interactive CLI chat
│   ├── chat_eval.py                # Task evaluation
│   ├── chat_rl.py                  # Reinforcement learning
│   ├── chat_sft.py                 # Supervised fine-tuning
│   ├── infer_bench.py              # Inference benchmarking
│   ├── tok_eval.py                 # Tokenizer evaluation
│   └── tok_train.py                # Tokenizer training
├── tasks/                          # Evaluation datasets & benchmarks (MMLU, GSM8K, ARC, HumanEval)
└── tests/                          # Pytest test suite
```

---

## 🤝 Contributing

The goal of this project is to push the frontier of accessible micro-models (<$1000 budget) without cognitive complexity. No giant configuration factories or if-then monsters — just clean, readable, hackable code. Pull requests and feedback are welcome!

---

## 📜 Acknowledgements & Citation

Based on [nanochat](https://github.com/karpathy/nanochat) by **Andrej Karpathy**, with inspiration from [nanoGPT](https://github.com/karpathy/nanoGPT) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 📄 License

MIT
