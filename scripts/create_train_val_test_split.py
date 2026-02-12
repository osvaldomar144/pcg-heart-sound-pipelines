from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")


def discover_classes(input_dir: Path) -> list[Path]:
    class_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if len(class_dirs) != 2:
        raise ValueError(
            f"La directory di input deve contenere esattamente 2 classi. Trovate: {len(class_dirs)}"
        )
    return class_dirs


def collect_files(class_dir: Path) -> list[Path]:
    files = [p for p in class_dir.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"Nessun file trovato nella classe: {class_dir}")
    return sorted(files)


def compute_counts(
    n_samples: int, train_frac: float, val_frac: float, test_frac: float
) -> dict[str, int]:
    exact = {
        "train": n_samples * train_frac,
        "val": n_samples * val_frac,
        "test": n_samples * test_frac,
    }
    counts = {k: int(math.floor(v)) for k, v in exact.items()}
    missing = n_samples - sum(counts.values())

    # Assegna eventuali campioni residui agli split con parte decimale maggiore.
    priorities = sorted(
        SPLITS,
        key=lambda k: (exact[k] - counts[k], {"train": 0, "val": 1, "test": 2}[k]),
        reverse=True,
    )
    for idx in range(missing):
        counts[priorities[idx % len(priorities)]] += 1

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Crea uno split train/val/test da una directory con 2 classi, "
            "mantenendo separazione per classe."
        )
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory sorgente con 2 sottocartelle classe (es. healthy/ unhealthy/)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory destinazione in cui creare train/ val/ test/",
    )
    ap.add_argument("--train_frac", type=float, default=0.75, help="Frazione train (default: 0.75i)")
    ap.add_argument("--val_frac", type=float, default=0.15, help="Frazione val (default: 0.15)")
    ap.add_argument("--test_frac", type=float, default=0.10, help="Frazione test (default: 0.10)")
    ap.add_argument("--seed", type=int, default=42, help="Seed random per split riproducibile")
    ap.add_argument(
        "--force",c
        action="store_true",
        help="Se out_dir esiste, lo cancella prima di creare lo split",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir non esiste: {input_dir}")

    for frac_name, frac in [
        ("train_frac", args.train_frac),
        ("val_frac", args.val_frac),
        ("test_frac", args.test_frac),
    ]:
        if frac < 0.0 or frac > 1.0:
            raise ValueError(f"{frac_name} deve essere tra 0 e 1. Valore ricevuto: {frac}")

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if not math.isclose(frac_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"Le frazioni devono sommare a 1.0. Somma ricevuta: {frac_sum:.8f}"
        )

    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} esiste già. Usa --force per sovrascrivere.")
        shutil.rmtree(out_dir)

    random.seed(args.seed)

    class_dirs = discover_classes(input_dir)
    summary: dict[str, dict[str, int]] = {split: {} for split in SPLITS}

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = collect_files(class_dir)
        random.shuffle(files)

        counts = compute_counts(
            n_samples=len(files),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
        i0 = counts["train"]
        i1 = i0 + counts["val"]
        split_files = {
            "train": files[:i0],
            "val": files[i0:i1],
            "test": files[i1:],
        }

        for split, file_list in split_files.items():
            summary[split][class_name] = len(file_list)
            for src in file_list:
                rel_path = src.relative_to(class_dir)
                dst = out_dir / split / class_name / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    print(f"Split creato in: {out_dir}")
    for split in SPLITS:
        split_total = sum(summary[split].values())
        print(f"\n[{split}] totale: {split_total}")
        for class_name, n_items in sorted(summary[split].items()):
            print(f"  {class_name}: {n_items}")


if __name__ == "__main__":
    main()
