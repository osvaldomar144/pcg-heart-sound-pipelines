from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


def build_transforms(img_size: int, pretrained_norm: bool) -> transforms.Compose:
    if pretrained_norm:
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


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def remap_dataset_to_checkpoint_classes(
    ds: datasets.ImageFolder, checkpoint_classes: list[str]
) -> datasets.ImageFolder:
    dataset_classes = ds.classes
    if set(dataset_classes) != set(checkpoint_classes):
        missing_in_test = sorted(set(checkpoint_classes) - set(dataset_classes))
        extra_in_test = sorted(set(dataset_classes) - set(checkpoint_classes))
        raise ValueError(
            "Classi incompatibili tra checkpoint e test set. "
            f"mancano_in_test={missing_in_test}, extra_in_test={extra_in_test}"
        )

    idx_map = {old_idx: checkpoint_classes.index(cls_name) for old_idx, cls_name in enumerate(dataset_classes)}
    ds.samples = [(path, idx_map[target]) for path, target in ds.samples]
    ds.imgs = ds.samples
    ds.targets = [idx_map[target] for target in ds.targets]
    ds.classes = checkpoint_classes
    ds.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(checkpoint_classes)}
    return ds


def plot_confusion(
    cm: np.ndarray,
    classes: list[str],
    out_path: Path,
    title: str,
    normalize: bool,
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums == 0, 1, row_sums)
        data = cm.astype(np.float64) / safe_row_sums
        fmt = ".2f"
        cmap = "Blues"
    else:
        data = cm
        fmt = "d"
        cmap = "Oranges"

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = data.max() / 2.0 if data.size else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = f"{value:{fmt}}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_global_metrics(metrics: dict[str, float], out_path: Path) -> None:
    names = list(metrics.keys())
    values = [metrics[name] for name in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=["#457B9D", "#E76F51", "#2A9D8F", "#F4A261"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Metriche globali sul test set")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_class_metrics(
    classes: list[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Metriche per classe")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for idx, cls_name in enumerate(classes):
        ax.text(x[idx], 1.02, f"n={int(support[idx])}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_stats_txt(
    out_path: Path,
    *,
    run_datetime: str,
    model_path: Path,
    test_dir: Path,
    out_dir: Path,
    device: str,
    img_size: int,
    pretrained_norm: bool,
    batch_size: int,
    num_workers: int,
    num_samples: int,
    classes: list[str],
    class_counts_true: list[int],
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    precision_weighted: float,
    recall_weighted: float,
    f1_weighted: float,
    precision_per_class: np.ndarray,
    recall_per_class: np.ndarray,
    f1_per_class: np.ndarray,
    support_per_class: np.ndarray,
    cm: np.ndarray,
    class_report: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"run_datetime: {run_datetime}\n")
        f.write(f"model_path: {model_path.resolve()}\n")
        f.write(f"test_dir: {test_dir.resolve()}\n")
        f.write(f"out_dir: {out_dir.resolve()}\n")
        f.write(f"device: {device}\n")
        f.write(f"img_size: {img_size}\n")
        f.write(f"pretrained_norm: {pretrained_norm}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"num_workers: {num_workers}\n")
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"classes: {classes}\n")
        f.write(f"class_counts_true: {class_counts_true}\n")
        f.write("\n")
        f.write(f"accuracy: {accuracy:.6f}\n")
        f.write(f"precision_macro: {precision_macro:.6f}\n")
        f.write(f"recall_macro: {recall_macro:.6f}\n")
        f.write(f"f1_macro: {f1_macro:.6f}\n")
        f.write(f"precision_weighted: {precision_weighted:.6f}\n")
        f.write(f"recall_weighted: {recall_weighted:.6f}\n")
        f.write(f"f1_weighted: {f1_weighted:.6f}\n")
        f.write("\nper_class_metrics:\n")
        for idx, cls_name in enumerate(classes):
            f.write(
                f"  {cls_name}: precision={precision_per_class[idx]:.6f} "
                f"recall={recall_per_class[idx]:.6f} f1={f1_per_class[idx]:.6f} "
                f"support={int(support_per_class[idx])}\n"
            )
        f.write("\nconfusion_matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nclassification_report:\n")
        f.write(class_report)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Valuta un checkpoint CNN su un test set ImageFolder e salva metriche + plot in runs/metrics."
        )
    )
    ap.add_argument("--model_path", required=True, help="Checkpoint .pt (es. model_best.pt)")
    ap.add_argument("--test_dir", required=True, help="Directory test con classi come sottocartelle")
    ap.add_argument("--out_dir", default="runs/metrics", help="Cartella base output metriche")
    ap.add_argument("--img_size", type=int, default=224, help="Resize immagini (default: 224)")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size inference")
    ap.add_argument("--num_workers", type=int, default=4, help="Num workers DataLoader")
    ap.add_argument(
        "--pretrained_norm",
        action="store_true",
        help="Usa normalizzazione ImageNet (attivala se in training hai usato --pretrained)",
    )
    args = ap.parse_args()

    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    out_base = Path(args.out_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir non trovata: {test_dir}")

    # Permette sia:
    # - path diretto a test/
    # - path root con train/val/test (in tal caso usa automaticamente test/)
    if (test_dir / "test").exists():
        test_dir = test_dir / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint non valido: manca la chiave 'model_state'")
    if "classes" not in checkpoint:
        raise KeyError("Checkpoint non valido: manca la chiave 'classes'")

    classes = checkpoint["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("La chiave 'classes' deve essere una lista non vuota")

    transform = build_transforms(img_size=args.img_size, pretrained_norm=args.pretrained_norm)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)
    test_ds = remap_dataset_to_checkpoint_classes(test_ds, checkpoint_classes=classes)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    if not y_true:
        raise RuntimeError(f"Nessun sample trovato in test_dir: {test_dir}")

    labels_idx = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0
    )
    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )

    class_report = classification_report(
        y_true,
        y_pred,
        labels=labels_idx,
        target_names=classes,
        zero_division=0,
    )

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_path.stem}_{test_dir.name}"
    run_dir = out_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion(
        cm=cm,
        classes=classes,
        out_path=run_dir / "confusion_matrix.png",
        title="Confusion Matrix (counts)",
        normalize=False,
    )
    plot_confusion(
        cm=cm,
        classes=classes,
        out_path=run_dir / "confusion_matrix_normalized.png",
        title="Confusion Matrix (normalized by true class)",
        normalize=True,
    )

    global_metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
    }
    plot_global_metrics(global_metrics, run_dir / "global_metrics.png")
    plot_per_class_metrics(
        classes=classes,
        precision=precision_cls,
        recall=recall_cls,
        f1=f1_cls,
        support=support_cls,
        out_path=run_dir / "per_class_metrics.png",
    )

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
                "classes": classes,
                "support_per_class": [int(x) for x in support_cls.tolist()],
                "precision_per_class": [float(x) for x in precision_cls.tolist()],
                "recall_per_class": [float(x) for x in recall_cls.tolist()],
                "f1_per_class": [float(x) for x in f1_cls.tolist()],
                "confusion_matrix": cm.tolist(),
                "num_samples": len(y_true),
            },
            f,
            indent=2,
        )

    save_stats_txt(
        out_path=run_dir / "stats.txt",
        run_datetime=datetime.now().isoformat(timespec="seconds"),
        model_path=model_path,
        test_dir=test_dir,
        out_dir=run_dir,
        device=str(device),
        img_size=args.img_size,
        pretrained_norm=args.pretrained_norm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=len(y_true),
        classes=classes,
        class_counts_true=[int(x) for x in np.bincount(np.array(y_true), minlength=len(classes)).tolist()],
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        precision_per_class=precision_cls,
        recall_per_class=recall_cls,
        f1_per_class=f1_cls,
        support_per_class=support_cls,
        cm=cm,
        class_report=class_report,
    )

    print("Evaluation completed.")
    print(f"Samples: {len(y_true)}")
    print(
        f"accuracy={accuracy:.4f} precision_macro={precision_macro:.4f} "
        f"recall_macro={recall_macro:.4f} f1_macro={f1_macro:.4f}"
    )
    print(f"Saved outputs in: {run_dir}")


if __name__ == "__main__":
    main()
