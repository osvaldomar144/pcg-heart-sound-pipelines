from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


def build_transforms(img_size: int, pretrained: bool) -> transforms.Compose:
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        try:
            mean = weights.meta["mean"]
            std = weights.meta["std"]
        except Exception:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root con train/ e val/ di PNG")
    ap.add_argument("--out_dir", default="runs/cnn", help="Cartella output per i checkpoint")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true", help="Usa pesi ImageNet")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs deve essere >= 1")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("data_dir deve contenere train/ e val/")

    tfm = build_transforms(args.img_size, args.pretrained)
    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds = datasets.ImageFolder(val_dir, transform=tfm)

    class_counts = torch.bincount(
        torch.tensor(train_ds.targets),
        minlength=len(train_ds.classes),
    )
    min_count = int(class_counts.min().item())
    max_count = int(class_counts.max().item())
    imbalance_ratio = float("inf") if min_count == 0 else max_count / min_count
    use_balancer = imbalance_ratio >= 1.5

    if use_balancer:
        # Usa una sola strategia di bilanciamento per evitare sovracorrezione:
        # qui scegliamo il campionamento pesato, quindi loss senza class weights.
        sampling_weights = 1.0 / class_counts.float().clamp(min=1.0)
        sample_weights = sampling_weights[torch.tensor(train_ds.targets)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=len(train_ds.classes), pretrained=args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    data_dir_name = Path(args.data_dir).name
    if data_dir_name.startswith("pipeline_"):
        pipeline_name = data_dir_name[len("pipeline_") :]
    else:
        pipeline_name = data_dir_name
    run_dir = Path(args.out_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pipeline_{pipeline_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "model_best.pt"
    last_ckpt_path = run_dir / "model_last.pt"

    stats_path = run_dir / "stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"run_datetime: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"data_dir: {Path(args.data_dir).resolve()}\n")
        f.write(f"pipeline: {pipeline_name}\n")
        f.write(f"device: {device}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"optimizer: {args.optimizer}\n")
        f.write(f"momentum: {args.momentum}\n")
        f.write(f"img_size: {args.img_size}\n")
        f.write(f"pretrained: {args.pretrained}\n")
        f.write(f"num_workers: {args.num_workers}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"train_samples: {len(train_ds)}\n")
        f.write(f"val_samples: {len(val_ds)}\n")
        f.write(f"classes: {train_ds.classes}\n")
        f.write(f"class_counts: {class_counts.tolist()}\n")
        f.write(f"imbalance_ratio: {imbalance_ratio:.3f}\n")
        f.write(f"balancer_enabled: {use_balancer}\n")

    best_acc = -1.0
    best_epoch = 0
    best_val_loss = 0.0
    last_train_loss = 0.0
    last_train_acc = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0
    completed_epochs = 0
    interrupted = False
    last_epoch_ckpt_path: Path | None = None

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
            last_train_loss = train_loss
            last_train_acc = train_acc
            last_val_loss = val_loss
            last_val_acc = val_acc

            print(
                f"epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": train_ds.classes,
                        "epoch": epoch,
                        "val_acc": val_acc,
                    },
                    best_ckpt_path,
                )

            epoch_ckpt_path = run_dir / f"model_epoch_{epoch:03d}.pt"
            epoch_state = {
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            torch.save(epoch_state, epoch_ckpt_path)
            torch.save(epoch_state, last_ckpt_path)
            completed_epochs = epoch
            last_epoch_ckpt_path = epoch_ckpt_path
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrotto da utente. Checkpoint conservati fino all'ultima epoca completata.")

    if completed_epochs == 0:
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "epoch": 0,
            },
            last_ckpt_path,
        )

    with open(stats_path, "a", encoding="utf-8") as f:
        f.write(f"completed_epochs: {completed_epochs}\n")
        f.write(f"interrupted: {interrupted}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        if best_epoch > 0:
            f.write(f"best_val_acc: {best_acc:.6f}\n")
            f.write(f"best_val_loss: {best_val_loss:.6f}\n")
        else:
            f.write("best_val_acc: n/a\n")
            f.write("best_val_loss: n/a\n")
        f.write(f"last_train_loss: {last_train_loss:.6f}\n")
        f.write(f"last_train_acc: {last_train_acc:.6f}\n")
        f.write(f"last_val_loss: {last_val_loss:.6f}\n")
        f.write(f"last_val_acc: {last_val_acc:.6f}\n")
        f.write(f"checkpoint_best: {best_ckpt_path}\n")
        f.write(f"checkpoint_last: {last_ckpt_path}\n")
        if last_epoch_ckpt_path is not None:
            f.write(f"checkpoint_last_epoch: {last_epoch_ckpt_path}\n")
        else:
            f.write("checkpoint_last_epoch: n/a\n")

    if best_ckpt_path.exists():
        print("Saved:", best_ckpt_path)
    else:
        print("Saved: no best checkpoint (nessuna epoca completata)")
    if last_epoch_ckpt_path is not None:
        print("Saved:", last_epoch_ckpt_path)
    print("Saved:", last_ckpt_path)


if __name__ == "__main__":
    main()
