from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

from src.data import FileItem, LABELS, discover_dataset
from src.pipelines import build_pipeline


LABEL_NAMES = {0: "healthy", 1: "unhealthy"}
DATASET_SPLITS = ["train", "val", "test"]


def discover_optional_test_split(data_dir: str) -> list[FileItem]:
    test_items: list[FileItem] = []
    test_root = Path(data_dir) / "test"
    if not test_root.exists():
        return test_items

    for cls_name, cls_label in LABELS.items():
        class_dir = test_root / cls_name
        if not class_dir.exists():
            continue
        for p in class_dir.glob("*.wav"):
            test_items.append(FileItem(path=str(p), label=cls_label, split="test"))
    return test_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Cartella dataset (contiene train/ e val/, opzionalmente test/)",
    )
    ap.add_argument("--cfg", default="configs/pipelines.json", help="Config con pipeline e parametri")
    ap.add_argument("--pipeline", default="bandpass_mel", help="Nome pipeline da applicare")
    ap.add_argument(
        "--out_dir",
        default="data_preprocessed",
        help="Cartella base per output (crea pipeline_{pipeline}_{datehour}/train|val|test/...)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive i file di output esistenti",
    )
    args = ap.parse_args()

    # Carica la configurazione delle pipeline (audio + TF + steps per pipeline).
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Scansiona train/val (obbligatori) e test (opzionale), poi costruisce la pipeline scelta.
    items = discover_dataset(args.data_dir)
    items.extend(discover_optional_test_split(args.data_dir))
    spec_fn = build_pipeline(cfg, args.pipeline)

    # Cartella di output: data_preprocessed/pipeline_{pipeline}_{datehour}/train|val|test/...
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_root = Path(args.out_dir) / f"pipeline_{args.pipeline}_{run_stamp}"

    total = 0
    skipped = 0
    counts = {}
    for item in items:
        # Mappa la label numerica al nome cartella e genera il path di output.
        cls_name = LABEL_NAMES.get(item.label, "unknown")
        in_path = Path(item.path)
        out_path = out_root / item.split / cls_name / f"{in_path.stem}.png"

        # Evita di riscrivere se il file esiste (a meno di --overwrite).
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # Applica la pipeline al file audio e salva lo spettrogramma.
        S = spec_fn(str(in_path))

        # Crea le cartelle necessarie e scrive il PNG.
        os.makedirs(out_path.parent, exist_ok=True)
        plt.imsave(out_path, S, cmap="gray", origin="lower")
        total += 1
        counts[(item.split, cls_name)] = counts.get((item.split, cls_name), 0) + 1

    # Report finale con conteggi.
    print(f"Saved {total} spectrograms to: {out_root}")
    if skipped:
        print(f"Skipped {skipped} existing files (use --overwrite to replace).")

    # Scrive stats.txt nella root dell'output della pipeline.
    os.makedirs(out_root, exist_ok=True)
    stats_path = out_root / "stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"run_datetime: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"pipeline: {args.pipeline}\n")
        f.write(f"run_stamp: {run_stamp}\n")
        f.write(f"processed_total: {total}\n")
        if skipped:
            f.write(f"skipped_existing: {skipped}\n")
        for split in DATASET_SPLITS:
            for cls_name in ["healthy", "unhealthy"]:
                key = (split, cls_name)
                f.write(f"{split}/{cls_name}: {counts.get(key, 0)}\n")


if __name__ == "__main__":
    main()
