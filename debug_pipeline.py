#!/usr/bin/env python3
"""Diagnostic: times each stage of the pipeline to find bottleneck."""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from pipeline_e.augmentation import AugmentationPipeline
from pipeline_e.config import PipelineEConfig
from pipeline_e.dataset import PipelineEDataset, make_collate_fn
from pipeline_e.model import WavLMMultiTaskModel
from shared.data_utils import load_commonvoice_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-csv", required=True, dest="cv_csv")
    parser.add_argument("--cv-audio-dir", required=True, dest="cv_audio_dir")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--num-workers", type=int, default=8, dest="num_workers")
    parser.add_argument("--n-batches", type=int, default=10, dest="n_batches")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Dataframe load
    t0 = time.time()
    df = load_commonvoice_df(args.cv_csv, args.cv_audio_dir)
    print(f"[{time.time()-t0:.1f}s] Loaded dataframe: {len(df)} rows")

    # 2) Single __getitem__ (no workers, no DataLoader)
    config = PipelineEConfig()
    augment = AugmentationPipeline(config)
    ds = PipelineEDataset(df.head(1000).reset_index(drop=True), augment=augment, training=True)
    t0 = time.time()
    for i in range(100):
        _ = ds[i]
    print(f"[__getitem__ solo] 100 samples: {time.time()-t0:.2f}s  ({(time.time()-t0)*10:.1f}ms each)")

    # 3) Model load
    t0 = time.time()
    model = WavLMMultiTaskModel(config).to(device)
    model.eval()
    print(f"[{time.time()-t0:.1f}s] Model loaded to {device}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    # 4) Feature extractor
    fe = AutoFeatureExtractor.from_pretrained(config.backbone_name)
    collate = make_collate_fn(fe)

    # 5) DataLoader with workers
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    t0 = time.time()
    it = iter(loader)
    first_batch = next(it)
    print(f"[first batch] {time.time()-t0:.2f}s  (includes worker boot)")

    # 6) Timed batches
    print(f"Timing {args.n_batches} batches through model...")
    times = {"data": [], "forward": [], "total": []}

    for i in range(args.n_batches):
        t_batch = time.time()
        batch = next(it)
        t_data = time.time()

        iv = batch["input_values"].to(device, non_blocking=True)
        am = batch.get("attention_mask")
        if am is not None:
            am = am.to(device, non_blocking=True)
        torch.cuda.synchronize()

        t_transfer = time.time()
        with torch.no_grad():
            out = model(iv, am)
        torch.cuda.synchronize()
        t_forward = time.time()

        times["data"].append(t_data - t_batch)
        times["forward"].append(t_forward - t_transfer)
        times["total"].append(t_forward - t_batch)
        print(f"  batch {i}: data={times['data'][-1]*1000:.0f}ms  "
              f"forward={times['forward'][-1]*1000:.0f}ms  "
              f"total={times['total'][-1]*1000:.0f}ms")

    print(f"\nMean: data={np.mean(times['data'])*1000:.0f}ms  "
          f"forward={np.mean(times['forward'])*1000:.0f}ms")
    print(f"Ratio: data is {np.mean(times['data'])/max(np.mean(times['forward']), 1e-6):.1f}x the forward time")


if __name__ == "__main__":
    main()
