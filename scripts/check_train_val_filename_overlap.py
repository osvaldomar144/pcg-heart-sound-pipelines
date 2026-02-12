from __future__ import annotations

import argparse
import sys
from pathlib import Path


def collect_by_name(root: Path) -> dict[str, list[Path]]:
    files_by_name: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        files_by_name.setdefault(p.name, []).append(p)
    return files_by_name


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verifica se esistono file con lo stesso nome sia in train che in val."
    )
    ap.add_argument("--data_dir", required=True, help="Directory dataset che contiene train/ e val/")
    ap.add_argument(
        "--max_print",
        type=int,
        default=50,
        help="Numero massimo di filename duplicati da stampare (default: 50)",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Struttura non valida: attesi {train_dir} e {val_dir}")

    train_files = collect_by_name(train_dir)
    val_files = collect_by_name(val_dir)

    overlap_names = sorted(set(train_files.keys()) & set(val_files.keys()))

    if not overlap_names:
        print("OK: nessun filename in comune tra train e val.")
        return

    print(f"Trovati {len(overlap_names)} filename presenti sia in train che in val.")
    print(f"Mostro i primi {min(len(overlap_names), args.max_print)}:")

    for name in overlap_names[: args.max_print]:
        print(f"\n{name}")
        for p in train_files[name]:
            print(f"  train: {p.relative_to(data_dir)}")
        for p in val_files[name]:
            print(f"  val:   {p.relative_to(data_dir)}")

    if len(overlap_names) > args.max_print:
        remaining = len(overlap_names) - args.max_print
        print(f"\n... altri {remaining} filename non mostrati.")

    sys.exit(1)


if __name__ == "__main__":
    main()
