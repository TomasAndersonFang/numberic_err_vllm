import argparse
import sys

from tqdm import tqdm
from evalplus.data import get_mbpp_plus
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils.io_utils import load_config, write_jsonl
from utils.prompt import format_prompt, extract_code


def run_vllm(tasks: list[dict], config: dict, tokenizer) -> list[dict]:
    """Run inference using vLLM backend on all tasks."""
    from vllm import LLM, SamplingParams

    model_name = config["model"]["name_or_path"]
    tp_size = config["model"]["tensor_parallel_size"]
    dtype = config["model"]["dtype"]
    gpu_memory_utilization = config["model"].get("gpu_memory_utilization", 0.9)
    enforce_eager = config["model"].get("enforce_eager", False)
    max_model_len = config["model"].get("max_model_len", None)
    temperature = config["generation"]["temperature"]
    max_new_tokens = config["generation"]["max_new_tokens"]

    print(f"Loading vLLM model: {model_name}")
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tp_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "max_num_batched_tokens": config["model"].get("max_num_batched_tokens", 2048),
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    print("Formatting prompts...")
    prompts = [format_prompt(task, tokenizer) for task in tqdm(tasks, desc="Prompts")]

    print("Running vLLM inference...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for task, output in tqdm(zip(tasks, outputs), total=len(tasks), desc="Extracting"):
        generated_text = output.outputs[0].text
        solution = extract_code(generated_text)
        results.append({"task_id": task["task_id"], "solution": solution})

    return results


def run_transformers(tasks: list[dict], config: dict) -> list[dict]:
    """Run inference using Transformers backend on all tasks."""
    model_name = config["model"]["name_or_path"]
    dtype_str = config["model"]["dtype"]
    temperature = config["generation"]["temperature"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    batch_size = config["generation"]["transformers_batch_size"]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"Loading Transformers model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch_dtype
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print("Formatting prompts...")
    prompts = [format_prompt(task, tokenizer) for task in tqdm(tasks, desc="Prompts")]

    results = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    print("Running Transformers inference...")
    for i in tqdm(range(num_batches), desc="Batches"):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_tasks = tasks[batch_start:batch_end]

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
            generated_ids = output_ids[j][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            solution = extract_code(generated_text)
            results.append({"task_id": task["task_id"], "solution": solution})

    return results


def main() -> None:
    """Main entry point for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["vllm", "transformers"])
    parser.add_argument("--model", default=None, help="Override model name_or_path")
    parser.add_argument("--dry-run", action="store_true", help="Run on first 10 tasks only")
    args = parser.parse_args()

    config = load_config("config.yaml")

    if args.model:
        config["model"]["name_or_path"] = args.model

    output_dir = config["output"]["dir"]
    output_path = f"{output_dir}/{args.backend}_samples.jsonl"

    print(f"Loading MBPP+ dataset...")
    dataset = get_mbpp_plus()
    tasks = []
    for task_id, task_data in dataset.items():
        task_entry = {"task_id": task_id, "prompt": task_data["prompt"]}
        tasks.append(task_entry)

    if args.dry_run:
        tasks = tasks[:10]
        print(f"Dry-run mode: using first {len(tasks)} tasks")

    print(f"Total tasks: {len(tasks)}")

    if args.backend == "vllm":
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name_or_path"])
        results = run_vllm(tasks, config, tokenizer)
    else:
        results = run_transformers(tasks, config)

    write_jsonl(output_path, results)
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
