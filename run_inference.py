import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from evalplus.data import get_mbpp_plus
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.io_utils import load_config, write_jsonl
from utils.prompt import format_prompt, extract_code


# Experiment modes for the layered attribution study described in the
# inference engine divergence experiment. Each mode pins a specific set of
# knobs so the five conditions stay reproducible.
#
#   A_transformers_bs1      Transformers, batch size 1, eager attention. The
#                           cleanest numerical reference path.
#   B_transformers_bsN      Transformers, batched (uses transformers_batch_size
#                           from config), eager attention. Adds left padding
#                           and batched matmul on top of A.
#   C_transformers_fa2      Transformers, batch size 1, flash_attention_2. On
#                           top of A, swaps the attention kernel to a fused
#                           implementation close to what vLLM uses.
#   D_vllm_serial           vLLM with max_num_seqs=1 and enforce_eager=True.
#                           Strips continuous batching and CUDA graph capture,
#                           leaving PagedAttention + KV cache paging as the
#                           main remaining differences vs Transformers.
#   E_vllm_default          vLLM with whatever the config specifies (your
#                           current production setting). The full stack.
EXPERIMENT_MODES = {
    "A_transformers_bs1",
    "B_transformers_bsN",
    "C_transformers_fa2",
    "D_vllm_serial",
    "E_vllm_default",
}


def set_seed(seed: int) -> None:
    """Fix RNG state across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_mode_settings(mode: str, config: dict) -> dict:
    """Translate an experiment mode into a concrete set of inference knobs.

    Returns a dict with keys describing which backend to use and the per-mode
    overrides applied on top of config.yaml.
    """
    if mode not in EXPERIMENT_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Valid modes: {sorted(EXPERIMENT_MODES)}"
        )

    if mode == "A_transformers_bs1":
        return {
            "backend": "transformers",
            "batch_size": 1,
            "attn_implementation": "eager",
        }
    if mode == "B_transformers_bsN":
        return {
            "backend": "transformers",
            "batch_size": config["generation"].get("transformers_batch_size", 8),
            "attn_implementation": "eager",
        }
    if mode == "C_transformers_fa2":
        return {
            "backend": "transformers",
            "batch_size": 1,
            "attn_implementation": "flash_attention_2",
        }
    if mode == "D_vllm_serial":
        return {
            "backend": "vllm",
            "max_num_seqs": 1,
            "enforce_eager": True,
        }
    # E_vllm_default
    return {
        "backend": "vllm",
        "max_num_seqs": config["model"].get("max_num_seqs", 256),
        "enforce_eager": config["model"].get("enforce_eager", False),
    }


def log_env(label: str) -> dict:
    """Capture environment details that may affect numerical results."""
    info = {
        "label": label,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        import vllm
        info["vllm_version"] = vllm.__version__
    except ImportError:
        pass
    try:
        import flash_attn
        info["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        info["flash_attn_version"] = None
    return info


def _fix_byte_level_literals(text: str, tokenizer) -> str:
    """Fallback reverse-map byte-level BPE literals if a tokenizer.decode
    output still contains Ġ (U+0120) / Ċ (U+010A) artifacts.

    This is only invoked when we detect the artifacts. In a healthy
    fast-tokenizer path this branch is a no-op.
    """
    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    if byte_decoder is None:
        # Slow tokenizers expose byte_decoder; fast ones usually do not.
        # As a last resort do a coarse character substitution.
        return text.replace("\u0120", " ").replace("\u010a", "\n")
    try:
        return bytearray(byte_decoder[c] for c in text).decode("utf-8", errors="replace")
    except KeyError:
        return text.replace("\u0120", " ").replace("\u010a", "\n")


def decode_completion(token_ids, tokenizer, fallback_text: str | None = None) -> str:
    """Decode generated token ids into a clean string.

    Strategy:
      1. If the caller supplies a fallback_text (typically vLLM's
         completion.text), trust it first. vLLM detokenizes through the
         same HF tokenizer and its output is canonical.
      2. Otherwise decode the token ids via the HF tokenizer.
      3. In either case, if the resulting string still contains byte-level
         BPE literals (Ġ / Ċ), reverse-map them.
    """
    if fallback_text is not None and fallback_text != "":
        text = fallback_text
    elif token_ids is None or len(token_ids) == 0:
        return ""
    else:
        text = tokenizer.decode(list(token_ids), skip_special_tokens=True)

    if "\u0120" in text or "\u010a" in text:
        text = _fix_byte_level_literals(text, tokenizer)

    return text


def run_vllm(tasks: list[dict], config: dict, mode_settings: dict, tokenizer) -> tuple[list[dict], list[dict]]:
    """Run inference using vLLM backend on all tasks."""
    from vllm import LLM, SamplingParams

    model_name = config["model"]["name_or_path"]
    tp_size = config["model"]["tensor_parallel_size"]
    dtype = config["model"]["dtype"]
    gpu_memory_utilization = config["model"].get("gpu_memory_utilization", 0.9)
    max_model_len = config["model"].get("max_model_len", None)
    temperature = config["generation"]["temperature"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    seed = config["generation"].get("seed", 42)

    # Mode specific overrides.
    enforce_eager = mode_settings["enforce_eager"]
    max_num_seqs = mode_settings["max_num_seqs"]

    print(f"Loading vLLM model: {model_name}")
    print(f"  enforce_eager={enforce_eager}, max_num_seqs={max_num_seqs}")

    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tp_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "max_num_batched_tokens": config["model"].get("max_num_batched_tokens", 2048),
        "max_num_seqs": max_num_seqs,
        "seed": seed,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    print("Formatting prompts...")
    prompts = [format_prompt(task, tokenizer) for task in tqdm(tasks, desc="Prompts")]

    print("Running vLLM inference...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    raw_results = []
    for task, prompt, output in tqdm(
        zip(tasks, prompts, outputs), total=len(tasks), desc="Extracting"
    ):
        completion = output.outputs[0]
        token_ids = list(completion.token_ids) if completion.token_ids is not None else []
        completion_text = getattr(completion, "text", None)

        # Trust vLLM's detokenized text; only fall back to manual decode if
        # vLLM did not provide one. If either path leaves byte-level BPE
        # literals in the string, decode_completion will reverse-map them.
        generated_text = decode_completion(
            token_ids, tokenizer, fallback_text=completion_text
        )

        solution = extract_code(generated_text)

        results.append({"task_id": task["task_id"], "solution": solution})
        raw_results.append({
            "task_id": task["task_id"],
            "prompt": prompt,
            "generated_text": generated_text,
            "token_ids": token_ids,
            "finish_reason": getattr(completion, "finish_reason", None),
            "solution": solution,
        })

    return results, raw_results


def run_transformers(tasks: list[dict], config: dict, mode_settings: dict, tokenizer) -> tuple[list[dict], list[dict]]:
    """Run inference using Transformers backend on all tasks."""
    model_name = config["model"]["name_or_path"]
    dtype_str = config["model"]["dtype"]
    temperature = config["generation"]["temperature"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    seed = config["generation"].get("seed", 42)

    batch_size = mode_settings["batch_size"]
    attn_implementation = mode_settings["attn_implementation"]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"Loading Transformers model: {model_name}")
    print(f"  batch_size={batch_size}, attn_implementation={attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("Formatting prompts...")
    prompts = [format_prompt(task, tokenizer) for task in tqdm(tasks, desc="Prompts")]

    results = []
    raw_results = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    print("Running Transformers inference...")
    for i in tqdm(range(num_batches), desc="Batches"):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_tasks = tasks[batch_start:batch_end]

        # When batch_size is 1 we skip padding entirely; the tokenizer still
        # supports the call but the resulting attention mask is all ones and
        # there is no left-pad region, which keeps the numerical path clean
        # for the A and C reference conditions.
        if batch_size == 1:
            inputs = tokenizer(batch_prompts, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature

            output_ids = model.generate(**inputs, **gen_kwargs)

        for j, task in enumerate(batch_tasks):
            input_len = inputs["input_ids"][j].shape[0]
            generated_ids = output_ids[j][input_len:].tolist()

            # Use the same decode helper as the vLLM path so both backends
            # produce strings through an identical decode route. Important
            # for the A vs D / B vs E text-level comparison.
            generated_text = decode_completion(generated_ids, tokenizer)

            solution = extract_code(generated_text)

            results.append({"task_id": task["task_id"], "solution": solution})
            raw_results.append({
                "task_id": task["task_id"],
                "prompt": batch_prompts[j],
                "generated_text": generated_text,
                "token_ids": generated_ids,
                "finish_reason": None,
                "solution": solution,
            })

    return results, raw_results


def main() -> None:
    """Main entry point for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(EXPERIMENT_MODES),
        help="Which attribution experiment condition to run.",
    )
    parser.add_argument("--model", default=None, help="Override model name_or_path")
    parser.add_argument("--dry-run", action="store_true", help="Run on first 10 tasks only")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["model"]["name_or_path"] = args.model

    seed = config["generation"].get("seed", 42)
    set_seed(seed)

    mode_settings = resolve_mode_settings(args.mode, config)
    backend = mode_settings["backend"]

    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.mode}_samples.jsonl"
    raw_output_path = f"{output_dir}/{args.mode}_raw.jsonl"
    meta_output_path = f"{output_dir}/{args.mode}_meta.json"

    print(f"Loading MBPP+ dataset...")
    dataset = get_mbpp_plus()
    tasks = []
    for task_id, task_data in dataset.items():
        task_entry = {"task_id": task_id, "prompt": task_data["prompt"]}
        tasks.append(task_entry)

    if args.dry_run:
        tasks = tasks[:10]
        print(f"Dry run mode: using first {len(tasks)} tasks")

    print(f"Mode: {args.mode}, backend: {backend}")
    print(f"Total tasks: {len(tasks)}")

    env_info = log_env(args.mode)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    # Load tokenizer once and pass it into both backends. This keeps the
    # prompt formatting and the decode path identical across A through E,
    # so any text-level differences we observe are attributable to the
    # inference engine and not to tokenizer configuration drift.
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
    print(f"Tokenizer: {type(tokenizer).__name__}, is_fast={tokenizer.is_fast}")

    t0 = time.time()
    if backend == "vllm":
        results, raw_results = run_vllm(tasks, config, mode_settings, tokenizer)
    else:
        results, raw_results = run_transformers(tasks, config, mode_settings, tokenizer)
    elapsed = time.time() - t0

    write_jsonl(output_path, results)
    write_jsonl(raw_output_path, raw_results)

    meta = {
        "mode": args.mode,
        "backend": backend,
        "mode_settings": mode_settings,
        "config": config,
        "env": env_info,
        "num_tasks": len(tasks),
        "elapsed_seconds": elapsed,
        "dry_run": args.dry_run,
    }
    with open(meta_output_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Samples written to: {output_path}")
    print(f"Raw outputs written to: {raw_output_path}")
    print(f"Metadata written to: {meta_output_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()