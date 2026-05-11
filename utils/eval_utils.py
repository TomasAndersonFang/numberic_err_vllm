import json


def parse_evalplus_output(result_json_path: str) -> dict[str, str]:
    """Parse evalplus JSON output into a flat task_id -> 'pass'/'fail' dict."""
    with open(result_json_path, "r") as f:
        data = json.load(f)

    eval_results = data["eval"]
    parsed = {}
    for task_id, results in eval_results.items():
        r = results[0]
        # MBPP+ pass requires both base and plus tests to pass
        passed = r.get("base_status") == "pass" and r.get("plus_status") == "pass"
        parsed[task_id] = "pass" if passed else "fail"

    return parsed
