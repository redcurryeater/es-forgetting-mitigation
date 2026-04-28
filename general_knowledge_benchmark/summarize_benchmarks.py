"""Summarize lm-evaluation-harness JSON outputs into a base-vs-tuned delta table.

Usage:
  python summarize_benchmarks.py \
    --results_dir bench_out/results \
    --labels base,beta0,beta1
"""

import argparse
import glob
import json
import os


PREFERRED_METRIC = {
    "hellaswag": "acc_norm,none",
    "race": "acc,none",
    "mmlu": "acc,none",
}


def load_results(results_dir, label):
    # lm-eval writes results_*.json under <output_path>/<label>/<model-name>/
    pattern = os.path.join(results_dir, label, "**", "results_*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # fallback: directly under label/
        pattern = os.path.join(results_dir, label, "results_*.json")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results JSON found for label '{label}' under {results_dir}")
    with open(files[-1]) as f:
        return json.load(f)


def pick_metric(task_results, task_name):
    metric_key = PREFERRED_METRIC.get(task_name)
    if metric_key and metric_key in task_results:
        return metric_key, task_results[metric_key]
    # fallback: first acc-like metric
    for k, v in task_results.items():
        if isinstance(v, (int, float)) and (k.startswith("acc") or k.startswith("exact_match")):
            return k, v
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--labels", default="base,beta0,beta1")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",")]
    runs = {lbl: load_results(args.results_dir, lbl) for lbl in labels}

    # collect all top-level tasks across runs (mmlu shows up as 'mmlu' aggregate
    # plus per-subject; we use the aggregate row when present)
    all_tasks = set()
    for r in runs.values():
        all_tasks.update(r.get("results", {}).keys())

    # focus on the headline tasks (skip per-subject MMLU rows for the table)
    headline = ["mmlu", "hellaswag", "race"]
    headline = [t for t in headline if t in all_tasks]

    base_label = labels[0]
    print(f"\n{'task':<12} {'metric':<14} " + " ".join(f"{l:>10}" for l in labels)
          + " " + " ".join(f"{'Δ '+l:>10}" for l in labels[1:]))
    print("-" * (12 + 1 + 14 + 1 + 11 * len(labels) + 11 * (len(labels) - 1)))

    for task in headline:
        metric_name, base_val = pick_metric(runs[base_label]["results"][task], task)
        if base_val is None:
            continue
        vals = []
        for lbl in labels:
            tr = runs[lbl]["results"].get(task, {})
            v = tr.get(metric_name, None)
            vals.append(v)
        deltas = [(v - base_val) if v is not None else None for v in vals[1:]]
        row = f"{task:<12} {metric_name:<14} " + " ".join(
            f"{v:>10.4f}" if v is not None else f"{'-':>10}" for v in vals
        ) + " " + " ".join(
            f"{d:>+10.4f}" if d is not None else f"{'-':>10}" for d in deltas
        )
        print(row)

    print("\n(Positive Δ = tuned model improves over base; negative Δ = forgetting.)")


if __name__ == "__main__":
    main()
