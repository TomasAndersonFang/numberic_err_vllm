import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.io_utils import load_config
from utils.eval_utils import parse_evalplus_output


def main() -> None:
    """Score inference outputs using EvalPlus."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["vllm", "transformers"])
    args = parser.parse_args()

    config = load_config("config.yaml")
    output_dir = config["output"]["dir"]
    samples_path = f"{output_dir}/{args.backend}_samples.jsonl"

    if not Path(samples_path).exists():
        print(f"Error: {samples_path} not found. Run inference first.")
        sys.exit(1)

    # evalplus writes: samples.replace(".jsonl", "_eval_results.json")
    eval_json_path = samples_path.replace(".jsonl", "_eval_results.json")

    print(f"Running EvalPlus evaluation on: {samples_path}")
    result = subprocess.run(
        ["uv", "run", "evalplus.evaluate", "--dataset", "mbpp",
         "--samples", samples_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"EvalPlus failed:\n{result.stderr}")
        sys.exit(1)

    print(result.stdout)

    if not Path(eval_json_path).exists():
        print(f"Error: Expected EvalPlus results at {eval_json_path} but file not found.")
        sys.exit(1)

    print(f"Parsing results from: {eval_json_path}")
    parsed = parse_evalplus_output(eval_json_path)

    # Save parsed results
    results_output_path = f"{output_dir}/{args.backend}_eval_results.json"
    with open(results_output_path, "w") as f:
        json.dump(parsed, f, indent=2)

    # Print summary
    total = len(parsed)
    pass_count = sum(1 for v in parsed.values() if v == "pass")
    fail_count = total - pass_count
    pass_rate = (pass_count / total * 100) if total > 0 else 0

    print(f"\n{'='*40}")
    print(f"Backend: {args.backend}")
    print(f"Total tasks: {total}")
    print(f"Pass: {pass_count}")
    print(f"Fail: {fail_count}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"{'='*40}")
    print(f"Results saved to: {results_output_path}")


if __name__ == "__main__":
    main()
