from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

IMG_SIZE = 224


def build_transforms(pretrained_norm: bool) -> transforms.Compose:
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
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Checkpoint .pt (model_best/model_last/model_epoch)")
    ap.add_argument("--input_path", required=True, help="Immagine campione (.png/.jpg/...)")
    ap.add_argument(
        "--pretrained_norm",
        action="store_true",
        help="Usa normalizzazione ImageNet (attivala se in training hai usato --pretrained)",
    )
    args = ap.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input non trovato: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    if "model_state" not in ckpt:
        raise KeyError("Checkpoint non valido: manca la chiave 'model_state'")
    if "classes" not in ckpt:
        raise KeyError("Checkpoint non valido: manca la chiave 'classes'")

    classes = ckpt["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("La chiave 'classes' deve essere una lista non vuota")

    model = build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tfm = build_transforms(pretrained_norm=args.pretrained_norm)
    image = Image.open(input_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        pred_idx = int(torch.argmax(probs).item())

    print(f"input: {input_path}")
    print(f"model: {model_path}")
    print(f"predicted_class: {classes[pred_idx]}")
    print(f"predicted_prob: {probs[pred_idx].item():.6f}")
    print("class_probabilities:")
    for idx, cls_name in enumerate(classes):
        print(f"  {cls_name}: {probs[idx].item():.6f}")


if __name__ == "__main__":
    main()
