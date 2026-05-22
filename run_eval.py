import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils.io_utils import load_config
from utils.eval_utils import parse_evalplus_output


# Keep in sync with EXPERIMENT_MODES in run_inference.py.
EXPERIMENT_MODES = [
    "A_transformers_bs1",
    "B_transformers_bsN",
    "C_transformers_fa2",
    "D_vllm_serial",
    "E_vllm_default",
]


def main() -> None:
    """Score inference outputs using EvalPlus."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=EXPERIMENT_MODES,
        help="Which attribution experiment condition to score.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["dir"]
    samples_path = f"{output_dir}/{args.mode}_samples.jsonl"

    if not Path(samples_path).exists():
        print(f"Error: {samples_path} not found. Run inference first.")
        sys.exit(1)

    # EvalPlus writes its raw results to: samples.replace(".jsonl", "_eval_results.json")
    # That lands at {mode}_samples_eval_results.json. We then parse and save our
    # own summary to {mode}_eval_parsed.json so the two files are easy to
    # distinguish at a glance when comparing five conditions.
    evalplus_raw_path = samples_path.replace(".jsonl", "_eval_results.json")

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

    if not Path(evalplus_raw_path).exists():
        print(
            f"Error: Expected EvalPlus results at {evalplus_raw_path} "
            "but file not found."
        )
        sys.exit(1)

    print(f"Parsing results from: {evalplus_raw_path}")
    parsed = parse_evalplus_output(evalplus_raw_path)

    # Summary stats.
    total = len(parsed)
    pass_count = sum(1 for v in parsed.values() if v == "pass")
    fail_count = total - pass_count
    pass_rate = (pass_count / total * 100) if total > 0 else 0

    # Save parsed results along with summary metadata so the comparison script
    # downstream can load a single file per mode without recomputing counts.
    parsed_output_path = f"{output_dir}/{args.mode}_eval_parsed.json"
    payload = {
        "mode": args.mode,
        "samples_path": samples_path,
        "evalplus_raw_path": evalplus_raw_path,
        "summary": {
            "total": total,
            "pass": pass_count,
            "fail": fail_count,
            "pass_rate": pass_rate,
        },
        "per_task": parsed,
    }
    with open(parsed_output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Mode: {args.mode}")
    print(f"Total tasks: {total}")
    print(f"Pass: {pass_count}")
    print(f"Fail: {fail_count}")
    print(f"Pass rate: {pass_rate:.1f}%")
    print(f"{'='*40}")
    print(f"Parsed results saved to: {parsed_output_path}")


if __name__ == "__main__":
    main()