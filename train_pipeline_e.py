#!/usr/bin/env python3
"""
Training script for Pipeline E (WavLM-large multi-task speaker profiling).

Smoke test (frozen, 2 epochs):
    python train_pipeline_e.py --mode frozen --epochs 2 --seed 0

Full frozen + LoRA:
    python train_pipeline_e.py --mode both --seed 0

Multi-seed:
    python train_pipeline_e.py --mode both --seeds 0 1 2 3 4
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from pipeline_e.augmentation import AugmentationPipeline
from pipeline_e.calibration import TemperatureScaler
from pipeline_e.config import PipelineEConfig
from pipeline_e.dataset import PipelineEDataset, make_collate_fn
from pipeline_e.losses import MultiTaskLoss
from pipeline_e.model import WavLMMultiTaskModel
from pipeline_e.sampler import build_weighted_sampler
from shared.data_utils import load_commonvoice_df, speaker_independent_split


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ordinal_logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=-1).long()


def age_logits_to_pred(logits: torch.Tensor, age_head_type: str) -> torch.Tensor:
    if age_head_type == "ordinal":
        return ordinal_logits_to_pred(logits)
    return logits.argmax(dim=-1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_age_true, all_age_pred = [], []
    all_logits, all_labels = [], []

    for batch in loader:
        iv = batch["input_values"].to(device)
        am = batch.get("attention_mask")
        if am is not None:
            am = am.to(device)
        age_lbl = batch["age_label"].to(device)
        gender_lbl = batch["gender_label"].to(device)

        out = model(iv, am)
        loss, _ = criterion(out["age_logits"], out["gender_logits"], age_lbl, gender_lbl)
        total_loss += loss.item()

        preds = age_logits_to_pred(out["age_logits"], config.age_head_type)
        all_age_true.extend(age_lbl.cpu().numpy())
        all_age_pred.extend(preds.cpu().numpy())
        all_logits.append(out["age_logits"].cpu())
        all_labels.append(age_lbl.cpu())

    macro_f1 = f1_score(all_age_true, all_age_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_age_true, all_age_pred)

    return {
        "loss": total_loss / max(len(loader), 1),
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
    }


def train_one_stage(
    model,
    train_loader,
    val_loader,
    config: PipelineEConfig,
    device: torch.device,
    stage_name: str,
    save_dir: str,
    lora_mode: bool = False,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    criterion = MultiTaskLoss(config).to(device)

    if lora_mode:
        backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config.learning_rate / 10},
            {"params": head_params, "lr": config.learning_rate},
        ], weight_decay=config.weight_decay)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            iv = batch["input_values"].to(device)
            am = batch.get("attention_mask")
            if am is not None:
                am = am.to(device)
            age_lbl = batch["age_label"].to(device)
            gender_lbl = batch["gender_label"].to(device)

            optimizer.zero_grad()
            out = model(iv, am)
            loss, _ = criterion(out["age_logits"], out["gender_logits"], age_lbl, gender_lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        val = evaluate(model, val_loader, criterion, device, config)

        print(
            f"[{stage_name}] Epoch {epoch:3d}/{config.num_epochs} "
            f"train={avg_loss:.4f} val_loss={val['loss']:.4f} "
            f"macro_f1={val['macro_f1']:.4f} bal_acc={val['balanced_acc']:.4f}"
        )

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_f1,
                "config": config,
            }, os.path.join(save_dir, "best_model.pt"))
        else:
            no_improve += 1
            if no_improve >= config.patience:
                print(f"  Early stop at epoch {epoch} (best={best_epoch}, f1={best_f1:.4f})")
                break

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if config.calibrate and config.age_head_type != "ordinal":
        val = evaluate(model, val_loader, criterion, device, config)
        scaler = TemperatureScaler()
        temp = scaler.fit(val["logits"], val["labels"])
        torch.save(scaler.state_dict(), os.path.join(save_dir, "calibrator.pt"))
        print(f"  Temperature scaling: T={temp:.4f}")

    print(f"  [{stage_name}] Best macro-F1={best_f1:.4f} @ epoch {best_epoch}")
    return {"macro_f1": best_f1, "balanced_acc": val.get("balanced_acc", 0.0)}


def build_loaders(config: PipelineEConfig, seed: int):
    print("Loading Common Voice AU data...")
    df = load_commonvoice_df(config.cv_csv, config.cv_audio_dir)
    print(f"  {len(df)} labeled samples with valid age/gender/audio")

    train_idx, val_idx, _ = speaker_independent_split(df, config.train_frac, config.val_frac, seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(config.backbone_name)
    augment = AugmentationPipeline(config)
    collate = make_collate_fn(feature_extractor)

    sampler = build_weighted_sampler(train_df, stratify_joint=True)
    train_ds = PipelineEDataset(train_df, augment=augment, training=True)
    val_ds = PipelineEDataset(val_df, augment=augment, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        collate_fn=collate, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate, num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


def run_training(args, seed: int) -> dict:
    set_seed(seed)
    device = get_device()
    print(f"\n=== Seed {seed} | device={device} ===")

    config_kwargs = dict(
        backbone_name=args.backbone,
        age_head_type=args.age_head,
        use_gender_moe=not args.no_moe,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        calibrate=not args.no_calibrate,
        use_augmentation=not args.no_augment,
        seed=seed,
        output_dir=args.output_dir,
    )
    if args.cv_csv is not None:
        config_kwargs["cv_csv"] = args.cv_csv
    if args.cv_audio_dir is not None:
        config_kwargs["cv_audio_dir"] = args.cv_audio_dir
    config = PipelineEConfig(**config_kwargs)
    if "base" in config.backbone_name:
        config.embedding_dim = 768

    train_loader, val_loader = build_loaders(config, seed)
    stage_metrics = {}

    if args.mode in ("frozen", "both"):
        print("\n--- Stage 1: Frozen backbone ---")
        frozen_dir = os.path.join(config.output_dir, f"seed_{seed}", "frozen")
        config.freeze_backbone = True
        config.use_lora = False
        model = WavLMMultiTaskModel(config).to(device)
        stage_metrics["frozen"] = train_one_stage(
            model, train_loader, val_loader, config, device, "Frozen", frozen_dir, lora_mode=False
        )

    if args.mode in ("lora", "both"):
        print("\n--- Stage 2: LoRA fine-tuning ---")
        lora_dir = os.path.join(config.output_dir, f"seed_{seed}", "lora")
        config.freeze_backbone = False
        config.use_lora = True
        model_lora = WavLMMultiTaskModel(config).to(device)

        if args.mode == "both":
            frozen_ckpt_path = os.path.join(config.output_dir, f"seed_{seed}", "frozen", "best_model.pt")
            frozen_state = torch.load(frozen_ckpt_path, map_location=device)["model_state"]
            model_dict = model_lora.state_dict()
            matched = {k: v for k, v in frozen_state.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched)
            model_lora.load_state_dict(model_dict, strict=False)
            print(f"  Loaded {len(matched)}/{len(model_dict)} weights from frozen stage")

        stage_metrics["lora"] = train_one_stage(
            model_lora, train_loader, val_loader, config, device, "LoRA", lora_dir, lora_mode=True
        )

    return stage_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Pipeline E")
    parser.add_argument("--backbone", default="microsoft/wavlm-large")
    parser.add_argument("--mode", choices=["frozen", "lora", "both"], default="both")
    parser.add_argument("--age-head", choices=["ldl", "ordinal", "softmax"], default="ldl", dest="age_head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--output-dir", default="pipeline_e_checkpoints", dest="output_dir")
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--cv-csv", default=None, dest="cv_csv")
    parser.add_argument("--cv-audio-dir", default=None, dest="cv_audio_dir")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_training(args, seed)

    if len(seeds) > 1:
        print("\n=== Multi-seed summary ===")
        for stage in ("frozen", "lora"):
            f1s = [r[stage]["macro_f1"] for r in all_results.values() if stage in r]
            if f1s:
                print(f"  [{stage}] macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSummary saved to {args.output_dir}/training_summary.json")


if __name__ == "__main__":
    main()
