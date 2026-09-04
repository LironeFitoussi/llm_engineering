# Week 3 Day 4: native Python version

[`day4_native.py`](day4_native.py) runs the Day 4 Hugging Face examples on a
local machine without `google.colab`, Colab Secrets, hard-coded CUDA calls, or
the notebook's CUDA-oriented 4-bit configuration.

## 1. Authenticate with Hugging Face

The recommended local setup is a one-time browser login:

```bash
uv run hf auth login
```

Alternatively, add a token to the `.env` file in the repository root:

```dotenv
HF_TOKEN=hf_your_token_here
```

Never put the token in the Python script or pass it as a command-line argument.
A read or appropriately scoped fine-grained token is sufficient for downloading
models.

Llama and Gemma are gated. Before using those aliases, request or accept access
with the same Hugging Face account:

- <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>
- <https://huggingface.co/google/gemma-3-270m-it>

## 2. Check the environment

From the repository root:

```bash
uv run python week3/day4_native.py --check
uv run python week3/day4_native.py --list-models
```

The script chooses CUDA, Apple MPS, or CPU in that order. Override it with
`--device mps` or `--device cpu` when needed.

## 3. Generate text

Start with the relatively small, ungated DeepSeek model:

```bash
uv run python week3/day4_native.py \
  --model deepseek \
  --prompt "Tell a light-hearted joke for a room of Data Scientists"
```

Other lesson aliases are `llama`, `phi`, `gemma`, and `qwen`. A full Hugging
Face model ID is also accepted:

```bash
uv run python week3/day4_native.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --max-new-tokens 80
```

The first run downloads model files into the Hugging Face cache. Models can be
several gigabytes, but later runs reuse the cached files. Only one model is
loaded per invocation, and the script releases its Python and accelerator
memory when generation finishes.
