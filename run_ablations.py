#!/usr/bin/env python3
"""
Launch all Pipeline E ablation variants and collect results.

Usage:
    python run_ablations.py [--epochs 30] [--seed 0]
"""

import argparse
import json
import os
import subprocess
import sys


ABLATIONS = [
    {
        "name": "frozen_ldl_moe",
        "mode": "frozen", "age_head": "ldl",
        "extra_flags": [],
    },
    {
        "name": "lora_ldl_moe",
        "mode": "lora", "age_head": "ldl",
        "extra_flags": [],
    },
    {
        "name": "frozen_softmax",
        "mode": "frozen", "age_head": "softmax",
        "extra_flags": [],
    },
    {
        "name": "frozen_ordinal",
        "mode": "frozen", "age_head": "ordinal",
        "extra_flags": [],
    },
    {
        "name": "frozen_nomoe",
        "mode": "frozen", "age_head": "ldl",
        "extra_flags": ["--no-moe"],
    },
    {
        "name": "frozen_nocal",
        "mode": "frozen", "age_head": "ldl",
        "extra_flags": ["--no-calibrate"],
    },
    {
        "name": "frozen_noaug",
        "mode": "frozen", "age_head": "ldl",
        "extra_flags": ["--no-augment"],
    },
]


def run_ablation(abl: dict, args) -> dict | None:
    name = abl["name"]
    out_dir = os.path.join(args.output_base, name)
    cmd = [
        sys.executable, "train_pipeline_e.py",
        "--mode", abl["mode"],
        "--age-head", abl["age_head"],
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--output-dir", out_dir,
        *abl["extra_flags"],
    ]
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"Command:  {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] {name} exited with code {result.returncode}")
        return None

    summary_path = os.path.join(out_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-base", default="pipeline_e_checkpoints", dest="output_base")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Run only these ablation names (space-separated)")
    args = parser.parse_args()

    ablations_to_run = ABLATIONS
    if args.only:
        ablations_to_run = [a for a in ABLATIONS if a["name"] in args.only]
        if not ablations_to_run:
            print(f"No ablations matched: {args.only}")
            print(f"Available: {[a['name'] for a in ABLATIONS]}")
            return

    all_results = {}
    for abl in ablations_to_run:
        result = run_ablation(abl, args)
        all_results[abl["name"]] = result

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    header = f"{'Ablation':<25} {'Stage':<8} {'macro_f1':>10} {'bal_acc':>10}"
    print(header)
    print("-" * 60)
    for name, result in all_results.items():
        if result is None:
            print(f"  {name:<25}  FAILED")
            continue
        for seed_key, stages in result.items():
            for stage, metrics in stages.items():
                f1 = metrics.get("macro_f1", float("nan"))
                acc = metrics.get("balanced_acc", float("nan"))
                print(f"  {name:<25} {stage:<8} {f1:>10.4f} {acc:>10.4f}")

    # ── Save ablation table ────────────────────────────────────────────────────
    rows = []
    for name, result in all_results.items():
        if result is None:
            continue
        for seed_key, stages in result.items():
            for stage, metrics in stages.items():
                rows.append({
                    "ablation": name,
                    "stage": stage,
                    "seed": seed_key,
                    "macro_f1": metrics.get("macro_f1"),
                    "balanced_acc": metrics.get("balanced_acc"),
                })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.output_base, "ablation_table.csv")
        df.to_csv(out_path, index=False)
        print(f"\nAblation table saved to {out_path}")


if __name__ == "__main__":
    main()
