#!/usr/bin/env python3
"""
Pre-decode audio files (MP3/WAV) to numpy .npy caches alongside each file.
One-time step to eliminate the MP3 decoding bottleneck during training.

Usage:
    python cache_audio_npy.py \
        --cv-csv /content/cv-data/cv-corpus-25.0-2026-03-09/en/commonvoice_subset.csv \
        --cv-audio-dir /content/cv-data/cv-corpus-25.0-2026-03-09/en/clips \
        --workers 12

For each audio file at `<dir>/<name>.mp3`, writes `<dir>/<name>.mp3.npy`
containing the decoded waveform at 16 kHz mono float32.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import librosa

TARGET_SR = 16000


def decode_one(audio_path: str, overwrite: bool = False) -> str:
    """Decode a single file → .npy. Returns status string."""
    npy_path = audio_path + ".npy"
    if not overwrite and os.path.exists(npy_path):
        return "skip"
    try:
        y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        np.save(npy_path, y.astype(np.float32))
        return "ok"
    except Exception as e:
        return f"err:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-csv", required=True, dest="cv_csv")
    parser.add_argument("--cv-audio-dir", required=True, dest="cv_audio_dir")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.cv_csv}...")
    df = pd.read_csv(args.cv_csv, low_memory=False)
    print(f"  {len(df)} rows")

    paths = [os.path.join(args.cv_audio_dir, p) for p in df["path"].tolist()]
    paths = [p for p in paths if os.path.exists(p)]
    print(f"  {len(paths)} files exist on disk")

    if not args.overwrite:
        to_do = [p for p in paths if not os.path.exists(p + ".npy")]
        print(f"  {len(to_do)} need decoding ({len(paths) - len(to_do)} already cached)")
    else:
        to_do = paths
        print(f"  {len(to_do)} to re-decode (overwrite=True)")

    if not to_do:
        print("Nothing to do.")
        return

    start = time.time()
    done = 0
    ok = 0
    err = 0

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(
            partial(decode_one, overwrite=args.overwrite),
            to_do,
            chunksize=32,
        ):
            done += 1
            if result == "ok":
                ok += 1
            elif result.startswith("err"):
                err += 1
            if done % 1000 == 0:
                elapsed = time.time() - start
                rate = done / max(elapsed, 1)
                eta = (len(to_do) - done) / max(rate, 1)
                sys.stdout.write(
                    f"\r  {done}/{len(to_do)}  ok={ok} err={err}  "
                    f"{rate:.0f} files/s  ETA {eta/60:.1f} min"
                )
                sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min. ok={ok} err={err}")


if __name__ == "__main__":
    main()
