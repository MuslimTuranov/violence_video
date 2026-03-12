import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ViolenceDataset, collect_video_paths
from model import get_model

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loaders(dataset_dir, batch_size, num_frames, num_workers, val_size, seed):
    samples = collect_video_paths(dataset_dir)
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_size,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    train_dataset = ViolenceDataset(train_samples, num_frames=num_frames)
    val_dataset = ViolenceDataset(val_samples, num_frames=num_frames)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, train_samples, val_samples

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses = []
    targets = []
    predictions = []
    progress = tqdm(loader, leave=False)
    for frames, labels, _ in progress:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(frames)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        losses.append(loss.item())
        targets.extend(labels.cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(targets, predictions),
        "report": classification_report(
            targets,
            predictions,
            target_names=["NonViolence", "Violence"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics

def save_artifacts(output_dir: Path, model, config, metrics, train_samples, val_samples):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_violence_model.safetensors"
    config_path = output_dir / "best_violence_model_config.json"
    metrics_path = output_dir / "best_val_metrics.json"
    split_path = output_dir / "data_split.json"

    save_file(model.state_dict(), str(model_path))
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    split_payload = {
        "train": [{"video_path": str(path), "label": label} for path, label in train_samples],
        "validation": [{"video_path": str(path), "label": label} for path, label in val_samples],
    }
    split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, train_samples, val_samples = build_loaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=args.seed,
    )

    model = get_model(
        num_classes=2,
        train_backbone=args.train_backbone,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_val_acc = 0.0
    best_metrics = None
    epochs_without_improvement = 0

    config = {
        "architecture": "r3d_18",
        "weights": "R3D_18_Weights.DEFAULT" if not args.no_pretrained else "random_init",
        "num_classes": 2,
        "num_frames": args.num_frames,
        "frame_size": 112,
        "label_map": {"0": "NonViolence", "1": "Violence"},
    }

    print(f"Using device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        print(
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy'] * 100:.2f}%"
        )
        print(
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy'] * 100:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            save_artifacts(
                Path(args.output_dir),
                model,
                config,
                val_metrics,
                train_samples,
                val_samples,
            )
            epochs_without_improvement = 0
            print(f"Saved new best model with val_acc={best_val_acc * 100:.2f}%")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping:
                print("Early stopping triggered")
                break

    if best_metrics is None:
        raise RuntimeError("Training finished without producing validation metrics")

    print("\nBest validation metrics:")
    print(json.dumps(best_metrics["report"], indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="videodataset",
        help="Path to dataset root containing Violence/ and NonViolence/",
    )
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=4)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
