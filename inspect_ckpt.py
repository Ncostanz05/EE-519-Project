#!/usr/bin/env python3
"""Quick inspection of a Pipeline E checkpoint + calibrator (CPU-only, no WavLM needed)."""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        default="pipeline_e_checkpoints/seed_0/lora",
        help="Directory containing best_model.pt and optional calibrator.pt",
    )
    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
    cal_path = os.path.join(args.ckpt_dir, "calibrator.pt")

    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print()
    print(f"  Best epoch:     {ckpt.get('epoch', '?')}")
    print(f"  Val macro-F1:   {ckpt.get('val_macro_f1', float('nan')):.4f}")

    state = ckpt.get("model_state", {})
    total_params = sum(v.numel() for v in state.values())
    print(f"  State keys:     {len(state)}")
    print(f"  Total params:   {total_params/1e6:.1f}M")

    # Show the head param names (non-backbone) — confirms heads are populated
    head_keys = [k for k in state.keys() if "backbone" not in k]
    print(f"  Head keys ({len(head_keys)}):")
    for k in head_keys[:20]:
        print(f"    {k:<50} {tuple(state[k].shape)}")
    if len(head_keys) > 20:
        print(f"    ... and {len(head_keys) - 20} more")

    if os.path.exists(cal_path):
        print()
        print(f"Loading {cal_path}...")
        cal = torch.load(cal_path, map_location="cpu", weights_only=False)
        # TemperatureScaler stores a single scalar parameter
        for k, v in cal.items():
            print(f"  {k}: {v.item() if v.numel() == 1 else v}")
    else:
        print(f"\n(no calibrator.pt at {cal_path})")


if __name__ == "__main__":
    main()
