from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from aircursor.models.static_mlp import STATIC_CLASSES, StaticGestureMLP


@dataclass
class TrainingConfig:
    data_root: Path
    output_path: Path
    batch_size: int = 64
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0


class StaticGestureDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.label_to_idx: Dict[str, int] = {}

        if not root.exists():
            raise FileNotFoundError(f"数据目录 {root} 不存在。")

        for label_dir in sorted(root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            if label not in STATIC_CLASSES:
                print(f"[警告] 跳过未知标签目录: {label_dir}")
                continue
            label_idx = self.label_to_idx.setdefault(label, len(self.label_to_idx))

            files = sorted(
                [
                    p
                    for p in label_dir.glob("*")
                    if p.suffix.lower() in {".csv", ".npy"}
                ]
            )
            if not files:
                print(f"[提示] 标签 {label} 下未找到样本文件。")
                continue

            for file_path in files:
                feature = self._load_feature(file_path)
                if feature is None:
                    continue
                self.samples.append((feature, label_idx))

        if not self.samples:
            raise RuntimeError(f"在目录 {root} 下未找到任何有效样本。")

        print(f"已加载 {len(self.samples)} 个样本，标签映射：{self.label_to_idx}")

    def _load_feature(self, path: Path) -> Optional[np.ndarray]:
        try:
            if path.suffix.lower() == ".csv":
                data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
            else:
                data = np.load(path).astype(np.float32)
        except Exception as exc:
            print(f"[警告] 读取 {path} 失败：{exc}")
            return None

        if data.ndim == 2 and data.shape[1] == 3:
            feature = data.flatten()
        else:
            feature = data.reshape(-1)

        if feature.size != 63 and feature.size != 64:
            print(f"[警告] 文件 {path} 大小 {feature.size} 不符合预期，已跳过。")
            return None

        if feature.size == 63:
            feature = np.concatenate([feature, np.array([1.0], dtype=np.float32)], axis=0)
        return feature.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feature, label_idx = self.samples[idx]
        return torch.from_numpy(feature), label_idx


def build_dataloaders(dataset: StaticGestureDataset, cfg: TrainingConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    if cfg.val_split <= 0 or cfg.val_split >= 1:
        train_dataset = dataset
        val_dataset = None
    else:
        val_size = int(len(dataset) * cfg.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(cfg: TrainingConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset = StaticGestureDataset(cfg.data_root)
    train_loader, val_loader = build_dataloaders(dataset, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    input_dim = dataset[0][0].numel()
    model = StaticGestureMLP(input_dim=input_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()
            print(
                f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )
        else:
            print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    output_path = cfg.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    payload = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": dataset.label_to_idx,
        "input_dim": input_dim,
    }
    torch.save(payload, output_path)
    print(f"已保存模型权重到 {output_path}")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="训练静态手势 MLP 模型")
    parser.add_argument("--data-root", type=Path, default=Path("data/static"), help="静态手势数据集根目录")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("modelsweights/static_mlp.pth"),
        help="模型权重输出路径",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正则")
    parser.add_argument("--val-split", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers 数量")
    args = parser.parse_args()
    return TrainingConfig(
        data_root=args.data_root,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()

