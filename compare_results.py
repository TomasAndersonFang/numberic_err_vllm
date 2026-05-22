import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils.io_utils import load_config


# Keep in sync with EXPERIMENT_MODES in run_inference.py and run_eval.py.
EXPERIMENT_MODES = [
    "A_transformers_bs1",
    "B_transformers_bsN",
    "C_transformers_fa2",
    "D_vllm_serial",
    "E_vllm_default",
]

# The attribution chain. Each pair isolates one variable; the diff in pass
# rates and the disagreement samples are what the comparison report exposes.
# Layout:  (label, mode_a, mode_b, what_this_pair_isolates)
ATTRIBUTION_PAIRS = [
    ("A_vs_B", "A_transformers_bs1", "B_transformers_bsN",
     "Padding and batched matmul (within Transformers)"),
    ("A_vs_C", "A_transformers_bs1", "C_transformers_fa2",
     "Attention kernel: eager vs FlashAttention 2"),
    ("C_vs_D", "C_transformers_fa2", "D_vllm_serial",
     "PagedAttention and vLLM custom kernels (serial)"),
    ("D_vs_E", "D_vllm_serial", "E_vllm_default",
     "Continuous batching and CUDA graph scheduling"),
    ("A_vs_E", "A_transformers_bs1", "E_vllm_default",
     "End-to-end: cleanest reference vs production vLLM"),
]


def load_parsed(output_dir: str, mode: str) -> dict:
    """Load a single mode's parsed eval json. Falls back to legacy filenames."""
    primary = Path(output_dir) / f"{mode}_eval_parsed.json"
    legacy = Path(output_dir) / f"{mode}_eval_results.json"
    path = primary if primary.exists() else legacy
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # Normalize: new format has a per_task field, old format is flat dict.
    if isinstance(data, dict) and "per_task" in data:
        return data
    return {
        "mode": mode,
        "summary": None,
        "per_task": data,
    }


def load_raw(output_dir: str, mode: str) -> dict:
    """Load generated text per task_id from {mode}_raw.jsonl if present."""
    path = Path(output_dir) / f"{mode}_raw.jsonl"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["task_id"]] = row.get("generated_text", "")
    return out


def truncate(text: str, limit: int = 2000) -> str:
    """Excel cells have a 32k character limit; trim long generations defensively."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def build_summary_rows(per_mode: dict, modes: list[str]) -> list[dict]:
    """One row per mode with pass count, fail count, and pass rate."""
    rows = []
    for mode in modes:
        data = per_mode.get(mode)
        if data is None:
            rows.append({
                "Mode": mode, "Status": "missing",
                "Total": "-", "Pass": "-", "Fail": "-", "Pass Rate": "-",
            })
            continue
        per_task = data["per_task"]
        total = len(per_task)
        pass_count = sum(1 for v in per_task.values() if v == "pass")
        fail_count = total - pass_count
        rate = (pass_count / total * 100) if total > 0 else 0
        rows.append({
            "Mode": mode, "Status": "ok",
            "Total": total, "Pass": pass_count, "Fail": fail_count,
            "Pass Rate": f"{rate:.2f}%",
        })
    return rows


def compute_pair_stats(
    mode_a: str, mode_b: str, per_mode: dict,
) -> dict | None:
    """Compute pass rates, delta, and disagreement counts for one pair."""
    data_a = per_mode.get(mode_a)
    data_b = per_mode.get(mode_b)
    if data_a is None or data_b is None:
        return None
    pt_a = data_a["per_task"]
    pt_b = data_b["per_task"]
    common = sorted(set(pt_a) & set(pt_b))
    total = len(common)
    if total == 0:
        return None

    pass_a = sum(1 for tid in common if pt_a[tid] == "pass")
    pass_b = sum(1 for tid in common if pt_b[tid] == "pass")
    only_a = sum(1 for tid in common if pt_a[tid] == "pass" and pt_b[tid] == "fail")
    only_b = sum(1 for tid in common if pt_a[tid] == "fail" and pt_b[tid] == "pass")
    agree = sum(1 for tid in common if pt_a[tid] == pt_b[tid])

    rate_a = pass_a / total * 100
    rate_b = pass_b / total * 100
    return {
        "mode_a": mode_a, "mode_b": mode_b,
        "total": total,
        "pass_a": pass_a, "pass_b": pass_b,
        "rate_a": rate_a, "rate_b": rate_b,
        "delta": rate_b - rate_a,
        "only_a": only_a, "only_b": only_b,
        "agree": agree,
        "agree_rate": agree / total * 100,
    }


def build_attribution_chain_rows(per_mode: dict) -> list[dict]:
    """One row per pair in the predefined attribution chain."""
    rows = []
    for label, mode_a, mode_b, isolates in ATTRIBUTION_PAIRS:
        stats = compute_pair_stats(mode_a, mode_b, per_mode)
        if stats is None:
            rows.append({
                "Pair": label,
                "Isolates": isolates,
                "Status": "missing one or both modes",
                "Pass Rate A": "-", "Pass Rate B": "-",
                "Delta (B - A)": "-",
                "Agreement Rate": "-",
                "Only A Pass": "-", "Only B Pass": "-",
            })
            continue
        rows.append({
            "Pair": label,
            "Isolates": isolates,
            "Mode A": mode_a,
            "Mode B": mode_b,
            "Pass Rate A": f"{stats['rate_a']:.2f}%",
            "Pass Rate B": f"{stats['rate_b']:.2f}%",
            "Delta (B - A)": f"{stats['delta']:+.2f}pp",
            "Agreement Rate": f"{stats['agree_rate']:.2f}%",
            "Only A Pass": stats["only_a"],
            "Only B Pass": stats["only_b"],
            "Total Compared": stats["total"],
        })
    return rows


def build_disagreement_table(
    mode_a: str, mode_b: str,
    per_mode: dict, per_mode_raw: dict,
) -> pd.DataFrame:
    """Per task table of disagreements between two modes. No prompt column;
    prompts are identical across modes so they offer no signal here."""
    data_a = per_mode.get(mode_a)
    data_b = per_mode.get(mode_b)
    if data_a is None or data_b is None:
        return pd.DataFrame()

    pt_a = data_a["per_task"]
    pt_b = data_b["per_task"]
    raw_a = per_mode_raw.get(mode_a, {})
    raw_b = per_mode_raw.get(mode_b, {})

    rows = []
    for tid in sorted(set(pt_a) & set(pt_b)):
        if pt_a[tid] == pt_b[tid]:
            continue
        rows.append({
            "task_id": tid,
            f"{mode_a}_result": pt_a[tid],
            f"{mode_b}_result": pt_b[tid],
            f"{mode_a}_generated": truncate(raw_a.get(tid, "")),
            f"{mode_b}_generated": truncate(raw_b.get(tid, "")),
        })
    return pd.DataFrame(rows)


def build_per_task_matrix(per_mode: dict) -> pd.DataFrame:
    """One row per task, one column per mode."""
    all_ids = set()
    for mode in EXPERIMENT_MODES:
        data = per_mode.get(mode)
        if data is not None:
            all_ids.update(data["per_task"].keys())
    all_ids = sorted(all_ids)

    rows = []
    for tid in all_ids:
        row = {"task_id": tid}
        for mode in EXPERIMENT_MODES:
            data = per_mode.get(mode)
            if data is None:
                row[mode] = "missing"
            else:
                row[mode] = data["per_task"].get(tid, "n/a")
        results = [row[m] for m in EXPERIMENT_MODES if row[m] not in ("missing", "n/a")]
        row["all_agree"] = "yes" if len(set(results)) <= 1 else "no"
        rows.append(row)
    return pd.DataFrame(rows)


def run_chain_report(output_dir: str) -> None:
    """Full attribution-chain report across all five modes."""
    print("Loading parsed eval results for each mode...")
    per_mode = {}
    missing = []
    for mode in EXPERIMENT_MODES:
        data = load_parsed(output_dir, mode)
        if data is None:
            missing.append(mode)
        per_mode[mode] = data
    if missing:
        print(f"Warning: missing eval results for: {missing}")
        print("  These modes will appear as 'missing' in the report.")

    print("Loading raw generation text for each mode (if available)...")
    per_mode_raw = {mode: load_raw(output_dir, mode) for mode in EXPERIMENT_MODES}

    summary_rows = build_summary_rows(per_mode, EXPERIMENT_MODES)
    chain_rows = build_attribution_chain_rows(per_mode)
    matrix_df = build_per_task_matrix(per_mode)

    # Sanity check: ensure all available modes share the same task id set.
    available = [m for m in EXPERIMENT_MODES if per_mode.get(m) is not None]
    if len(available) >= 2:
        ref = set(per_mode[available[0]]["per_task"].keys())
        for m in available[1:]:
            other = set(per_mode[m]["per_task"].keys())
            if ref != other:
                print(
                    f"Warning: task id mismatch between {available[0]} and {m}: "
                    f"only in {available[0]}={len(ref - other)}, "
                    f"only in {m}={len(other - ref)}"
                )

    report_path = f"{output_dir}/comparison_report.xlsx"
    print(f"Writing report to: {report_path}")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(chain_rows).to_excel(
            writer, sheet_name="Attribution_Chain", index=False
        )
        matrix_df.to_excel(writer, sheet_name="Per_Task_Matrix", index=False)

        for label, mode_a, mode_b, _ in ATTRIBUTION_PAIRS:
            df = build_disagreement_table(mode_a, mode_b, per_mode, per_mode_raw)
            sheet_name = f"Disagree_{label}"[:31]
            if df.empty:
                pd.DataFrame([{"note": "no data or no disagreements"}]).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
            else:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n{'='*60}")
    print("Pass rates per mode")
    print(f"{'='*60}")
    for row in summary_rows:
        print(
            f"  {row['Mode']:<25} {row['Status']:<10} "
            f"pass={row['Pass']} / {row['Total']}  rate={row['Pass Rate']}"
        )

    print(f"\n{'='*60}")
    print("Attribution chain (delta = mode B pass rate minus mode A)")
    print(f"{'='*60}")
    for row in chain_rows:
        if row.get("Status") == "missing one or both modes":
            print(f"  {row['Pair']:<10} [missing]  isolates: {row['Isolates']}")
            continue
        print(
            f"  {row['Pair']:<10} delta={row['Delta (B - A)']:<10} "
            f"agree={row['Agreement Rate']:<8} "
            f"onlyA={row['Only A Pass']:<4} onlyB={row['Only B Pass']:<4} "
            f"-- {row['Isolates']}"
        )
    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")


def run_pair_report(output_dir: str, mode_a: str, mode_b: str) -> None:
    """Single ad-hoc comparison between two arbitrary modes."""
    if mode_a == mode_b:
        print(f"Error: --pair requires two distinct modes, got {mode_a} twice.")
        sys.exit(1)

    print(f"Loading parsed eval results for {mode_a} and {mode_b}...")
    per_mode = {
        mode_a: load_parsed(output_dir, mode_a),
        mode_b: load_parsed(output_dir, mode_b),
    }
    for m in (mode_a, mode_b):
        if per_mode[m] is None:
            print(f"Error: parsed eval file for {m} not found in {output_dir}.")
            sys.exit(1)

    print("Loading raw generation text...")
    per_mode_raw = {
        mode_a: load_raw(output_dir, mode_a),
        mode_b: load_raw(output_dir, mode_b),
    }

    stats = compute_pair_stats(mode_a, mode_b, per_mode)
    if stats is None:
        print("Error: no overlapping task ids between the two modes.")
        sys.exit(1)

    summary_rows = build_summary_rows(per_mode, [mode_a, mode_b])
    pair_row = {
        "Mode A": mode_a,
        "Mode B": mode_b,
        "Pass Rate A": f"{stats['rate_a']:.2f}%",
        "Pass Rate B": f"{stats['rate_b']:.2f}%",
        "Delta (B - A)": f"{stats['delta']:+.2f}pp",
        "Agreement Rate": f"{stats['agree_rate']:.2f}%",
        "Only A Pass": stats["only_a"],
        "Only B Pass": stats["only_b"],
        "Total Compared": stats["total"],
    }
    disagree_df = build_disagreement_table(mode_a, mode_b, per_mode, per_mode_raw)

    # File name encodes the pair so multiple ad-hoc reports do not overwrite
    # each other or the main chain report.
    safe_a = mode_a.replace("/", "_")
    safe_b = mode_b.replace("/", "_")
    report_path = f"{output_dir}/comparison_{safe_a}__vs__{safe_b}.xlsx"
    print(f"Writing pair report to: {report_path}")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame([pair_row]).to_excel(
            writer, sheet_name="Pair_Comparison", index=False
        )
        if disagree_df.empty:
            pd.DataFrame([{"note": "no disagreements between the two modes"}]).to_excel(
                writer, sheet_name="Disagreements", index=False
            )
        else:
            disagree_df.to_excel(writer, sheet_name="Disagreements", index=False)

    print(f"\n{'='*60}")
    print(f"Pair comparison: {mode_a}  vs  {mode_b}")
    print(f"{'='*60}")
    print(f"  Pass rate A:       {stats['rate_a']:.2f}%  ({stats['pass_a']}/{stats['total']})")
    print(f"  Pass rate B:       {stats['rate_b']:.2f}%  ({stats['pass_b']}/{stats['total']})")
    print(f"  Delta (B - A):     {stats['delta']:+.2f}pp")
    print(f"  Agreement rate:    {stats['agree_rate']:.2f}%")
    print(f"  Only A pass:       {stats['only_a']}")
    print(f"  Only B pass:       {stats['only_b']}")
    print(f"{'='*60}")
    print(f"Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare inference modes. With no args, generates the full "
                    "attribution-chain report across all five modes. With --pair, "
                    "generates an ad-hoc two-mode comparison."
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("MODE_A", "MODE_B"),
        choices=EXPERIMENT_MODES,
        default=None,
        help="Run a single comparison between two specific modes.",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["dir"]

    if args.pair is None:
        run_chain_report(output_dir)
    else:
        mode_a, mode_b = args.pair
        run_pair_report(output_dir, mode_a, mode_b)


if __name__ == "__main__":
    main()