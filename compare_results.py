import json
import sys

import pandas as pd
from evalplus.data import get_mbpp_plus

from utils.io_utils import load_config


def main() -> None:
    """Generate comparison report between vLLM and Transformers eval results."""
    config = load_config("config.yaml")
    output_dir = config["output"]["dir"]

    vllm_path = f"{output_dir}/vllm_eval_results.json"
    transformers_path = f"{output_dir}/transformers_eval_results.json"

    print("Loading evaluation results...")
    with open(vllm_path, "r") as f:
        vllm_results = json.load(f)
    with open(transformers_path, "r") as f:
        transformers_results = json.load(f)

    # Assert same task IDs
    vllm_ids = set(vllm_results.keys())
    transformers_ids = set(transformers_results.keys())
    if vllm_ids != transformers_ids:
        missing_in_transformers = vllm_ids - transformers_ids
        missing_in_vllm = transformers_ids - vllm_ids
        print(f"Error: Task ID mismatch!")
        if missing_in_transformers:
            print(f"  Missing in transformers: {len(missing_in_transformers)}")
        if missing_in_vllm:
            print(f"  Missing in vllm: {len(missing_in_vllm)}")
        sys.exit(1)

    # Load MBPP+ for original prompts
    print("Loading MBPP+ dataset for prompts...")
    dataset = get_mbpp_plus()

    # Classify into buckets
    vllm_only_pass = []
    transformers_only_pass = []
    both_pass = []
    both_fail = []

    for task_id in sorted(vllm_results.keys()):
        vllm_res = vllm_results[task_id]
        trans_res = transformers_results[task_id]
        prompt = dataset[task_id]["prompt"] if task_id in dataset else ""

        row = {
            "task_id": task_id,
            "prompt": prompt,
            "vllm_result": vllm_res,
            "transformers_result": trans_res,
        }

        if vllm_res == "pass" and trans_res == "pass":
            both_pass.append(row)
        elif vllm_res == "pass" and trans_res == "fail":
            vllm_only_pass.append(row)
        elif vllm_res == "fail" and trans_res == "pass":
            transformers_only_pass.append(row)
        else:
            both_fail.append(row)

    total = len(vllm_results)
    agreement = len(both_pass) + len(both_fail)
    agreement_rate = (agreement / total * 100) if total > 0 else 0

    # Build summary
    summary_data = [
        {"Bucket": "both_pass", "Count": len(both_pass),
         "Percentage": f"{len(both_pass)/total*100:.1f}%"},
        {"Bucket": "both_fail", "Count": len(both_fail),
         "Percentage": f"{len(both_fail)/total*100:.1f}%"},
        {"Bucket": "vllm_only_pass", "Count": len(vllm_only_pass),
         "Percentage": f"{len(vllm_only_pass)/total*100:.1f}%"},
        {"Bucket": "transformers_only_pass", "Count": len(transformers_only_pass),
         "Percentage": f"{len(transformers_only_pass)/total*100:.1f}%"},
        {"Bucket": "---", "Count": "---", "Percentage": "---"},
        {"Bucket": "Total", "Count": total, "Percentage": "100.0%"},
        {"Bucket": "Agreement Rate", "Count": agreement,
         "Percentage": f"{agreement_rate:.1f}%"},
    ]

    # Export to Excel
    report_path = f"{output_dir}/comparison_report.xlsx"
    print(f"Writing report to: {report_path}")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(vllm_only_pass).to_excel(writer, sheet_name="vllm_only_pass", index=False)
        pd.DataFrame(transformers_only_pass).to_excel(
            writer, sheet_name="transformers_only_pass", index=False
        )
        pd.DataFrame(both_pass).to_excel(writer, sheet_name="both_pass", index=False)
        pd.DataFrame(both_fail).to_excel(writer, sheet_name="both_fail", index=False)

    # Print summary to stdout
    print(f"\n{'='*50}")
    print(f"Comparison Summary")
    print(f"{'='*50}")
    print(f"Total tasks:             {total}")
    print(f"Both pass:               {len(both_pass)} ({len(both_pass)/total*100:.1f}%)")
    print(f"Both fail:               {len(both_fail)} ({len(both_fail)/total*100:.1f}%)")
    print(f"vLLM only pass:          {len(vllm_only_pass)} ({len(vllm_only_pass)/total*100:.1f}%)")
    print(f"Transformers only pass:  {len(transformers_only_pass)} ({len(transformers_only_pass)/total*100:.1f}%)")
    print(f"Agreement rate:          {agreement_rate:.1f}%")
    print(f"{'='*50}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
