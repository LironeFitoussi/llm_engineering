#!/usr/bin/env python3
"""Run the Week 3 Day 4 Hugging Face examples without Google Colab.

The original notebook assumes a Colab T4, CUDA, bitsandbytes, and Colab
Secrets. This version uses the existing project environment, reads an
optional HF_TOKEN from the project-root .env file, and selects MPS, CUDA, or
CPU at runtime. It deliberately does not accept a token as a command-line
argument because command-line arguments can be retained in shell history.
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    description: str
    gated: bool = False


MODELS: dict[str, ModelSpec] = {
    "llama": ModelSpec(
        "meta-llama/Llama-3.2-1B-Instruct",
        "Llama 3.2 1B Instruct (small, but requires Meta approval)",
        gated=True,
    ),
    "phi": ModelSpec(
        "microsoft/Phi-4-mini-instruct",
        "Phi-4 Mini Instruct",
    ),
    "gemma": ModelSpec(
        "google/gemma-3-270m-it",
        "Gemma 3 270M Instruct (very small, but requires Google approval)",
        gated=True,
    ),
    "qwen": ModelSpec(
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen3 4B Instruct (largest local option in the lesson)",
    ),
    "deepseek": ModelSpec(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek R1 Distill Qwen 1.5B (recommended first local test)",
    ),
}


class Day4Error(RuntimeError):
    """An expected, user-actionable Day 4 error."""


def load_local_environment() -> bool:
    """Load the project .env without replacing variables already exported."""
    if not ENV_FILE.is_file():
        return False

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise Day4Error(
            "python-dotenv is missing. Run `uv sync` from the project root."
        ) from exc

    load_dotenv(ENV_FILE, override=False)
    return True


def resolve_model(value: str) -> tuple[str, ModelSpec | None]:
    """Resolve a friendly alias while still allowing any Hub model ID."""
    normalized = value.strip().lower()
    if normalized in MODELS:
        spec = MODELS[normalized]
        return spec.model_id, spec
    if "/" not in value:
        aliases = ", ".join(MODELS)
        raise Day4Error(
            f"Unknown model {value!r}. Use one of: {aliases}; or provide a Hub ID."
        )
    return value, None


def select_device(torch: Any, requested: str) -> str:
    """Select and validate the requested PyTorch execution device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise Day4Error("CUDA was requested, but no CUDA GPU is available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        reason = (
            "this PyTorch build has no MPS support"
            if not torch.backends.mps.is_built()
            else "MPS is not available to this Python process"
        )
        raise Day4Error(f"MPS was requested, but {reason}.")
    return requested


def select_dtype(torch: Any, device: str) -> Any:
    """Choose a conservative dtype for each backend."""
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def clear_memory(torch: Any, device: str) -> None:
    """Release Python and accelerator caches after a model run."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def token_is_available() -> bool:
    """Check for an environment or cached Hugging Face token without printing it."""
    try:
        from huggingface_hub import get_token
    except ImportError as exc:
        raise Day4Error(
            "huggingface-hub is missing. Run `uv sync` from the project root."
        ) from exc
    return bool(get_token())


def print_model_list() -> None:
    print("Available model aliases:\n")
    for alias, spec in MODELS.items():
        access = " [gated]" if spec.gated else ""
        print(f"  {alias:<9} {spec.model_id}{access}")
        print(f"            {spec.description}")


def run_check(requested_device: str) -> int:
    """Check the local setup without contacting the Hub or downloading a model."""
    try:
        import torch
        import transformers
        import huggingface_hub
    except ImportError as exc:
        raise Day4Error(
            f"A required package is missing ({exc.name}). Run `uv sync`."
        ) from exc

    device = select_device(torch, requested_device)
    dtype = select_dtype(torch, device)

    print("Week 3 Day 4 native environment")
    print(f"  Python:              {sys.version.split()[0]}")
    print(f"  PyTorch:             {torch.__version__}")
    print(f"  Transformers:        {transformers.__version__}")
    print(f"  huggingface-hub:     {huggingface_hub.__version__}")
    print(f"  Selected device:     {device}")
    print(f"  Model dtype:         {str(dtype).removeprefix('torch.')}")
    print(f"  Project .env found:  {'yes' if ENV_FILE.is_file() else 'no'}")
    print(f"  HF token available:  {'yes' if token_is_available() else 'no'}")

    if device == "cpu":
        print(
            "\nNote: no accelerator is visible to this process. The script will work, "
            "but generation may be slow."
        )
    return 0


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate(args: argparse.Namespace) -> int:
    """Load one model, generate one response, and then release its memory."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    except ImportError as exc:
        raise Day4Error(
            f"A required package is missing ({exc.name}). Run `uv sync`."
        ) from exc

    model_id, spec = resolve_model(args.model)
    if spec and spec.gated and not token_is_available():
        raise Day4Error(
            f"{model_id} is gated and no Hugging Face token was found. "
            "Run `uv run hf auth login`, then make sure your account has accepted "
            "the model's access terms."
        )

    device = select_device(torch, args.device)
    dtype = select_dtype(torch, device)
    dtype_name = str(dtype).removeprefix("torch.")

    print(f"Loading {model_id}", file=sys.stderr)
    print(f"Device: {device}; dtype: {dtype_name}", file=sys.stderr)
    print("The first run downloads and caches the model weights.\n", file=sys.stderr)

    torch.manual_seed(args.seed)
    model = None
    tokenizer = None
    model_inputs = None
    outputs = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.to(device)
        model.eval()

        messages = build_messages(args.prompt, args.system)
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = {name: tensor.to(device) for name, tensor in model_inputs.items()}

        generation_options: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if args.temperature > 0:
            generation_options.update(
                temperature=args.temperature,
                top_p=args.top_p,
            )

        if args.stream:
            generation_options["streamer"] = TextStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            with torch.inference_mode():
                model.generate(**generation_options)
        else:
            with torch.inference_mode():
                outputs = model.generate(**generation_options)
            prompt_length = model_inputs["input_ids"].shape[-1]
            answer_tokens = outputs[0][prompt_length:]
            print(tokenizer.decode(answer_tokens, skip_special_tokens=True))
    finally:
        del outputs, model_inputs, tokenizer, model
        clear_memory(torch, device)

    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Week 3 Day 4 Hugging Face models natively, without Colab."
        )
    )
    parser.add_argument(
        "--model",
        default="deepseek",
        help="Model alias or Hugging Face Hub ID (default: deepseek).",
    )
    parser.add_argument(
        "--prompt",
        default="Tell a light-hearted joke for a room of Data Scientists",
        help="User prompt sent to the model.",
    )
    parser.add_argument("--system", help="Optional system prompt.")
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Execution device (default: auto).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=120,
        help="Maximum generated tokens (default: 120).",
    )
    parser.add_argument(
        "--temperature",
        type=nonnegative_float,
        default=0.7,
        help="Sampling temperature; 0 selects deterministic decoding (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=probability,
        default=0.95,
        help="Nucleus sampling probability (default: 0.95).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Print the answer only after generation completes.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List lesson model aliases and exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check packages, device, and token presence without downloading a model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        load_local_environment()
        if args.list_models:
            print_model_list()
            return 0
        if args.check:
            return run_check(args.device)
        return generate(args)
    except Day4Error as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        print(f"Error: {message}", file=sys.stderr)
        print(
            "If this is a 401/403 error, check `uv run hf auth login` and accept "
            "the model's access terms. For more detail, rerun from a debugger.",
            file=sys.stderr,
        )
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
