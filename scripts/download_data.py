from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

# Slug Kaggle del dataset (usato dal comando: kaggle datasets download -d <slug>)
DATASET = "swapnilpanda/heart-sound-database"


def run(cmd: list[str]) -> None:
    """
    Esegue un comando di sistema (qui usato per chiamare la Kaggle CLI).
    - check=True: se il comando fallisce, solleva un'eccezione (così non proseguiamo con dati incompleti)
    """
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_train_val(root: Path) -> tuple[Path, Path]:
    """
    Cerca ricorsivamente dentro 'root' due directory chiamate 'train' e 'val'.
    - Kaggle può estrarre il dataset con una struttura di cartelle non sempre identica

    Ritorna:
    - (train_path, val_path)

    Se non le trova: raise FileNotFoundError, perché senza split train/val non possiamo proseguire.
    """
    train = None
    val = None

    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == "train":
            train = p
        if p.is_dir() and p.name.lower() == "val":
            val = p

    if train is None or val is None:
        raise FileNotFoundError(
            f"Non trovo 'train' e 'val' dentro {root}. "
            "Controlla cosa è stato estratto dal dataset."
        )

    return train, val


def main():
    """
    - Scarica il dataset da Kaggle usando Kaggle CLI
    - Estrae in una cartella temporanea
    - Copia le cartelle train/ e val/ dentro data_dir
    - Controlla che esista la struttura attesa: healthy/unhealthy
    - Stampa il numero di file .wav per classe (utile per capire sbilanciamento)

    Dipendenze:
    - Kaggle CLI installata (pip install kaggle)
    - Autenticazione Kaggle: kaggle.json in ~/.kaggle o env vars KAGGLE_USERNAME / KAGGLE_KEY
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Cartella destinazione (default: data/)")
    ap.add_argument("--dataset", default=DATASET, help="Slug Kaggle dataset")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Sovrascrive data/train e data/val se esistono (ATTENZIONE: cancella dati esistenti).",
    )
    args = ap.parse_args()

    # Percorsi base
    data_dir = Path(args.data_dir)

    # Cartella temporanea in cui scarichiamo e facciamo unzip
    # (poi la cancelliamo a fine esecuzione)
    tmp_dir = data_dir / "_tmp_download"

    # Creiamo le cartelle se non esistono
    tmp_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------
    # 1) Gestione caso in cui train/val esistano già
    # --------------------------------------
    # Se data/train o data/val esistono:
    # - senza --force: blocchiamo per evitare di sovrascrivere per errore
    # - con --force: li cancelliamo e riscarichiamo pulito
    for split in ["train", "val"]:
        dest = data_dir / split
        if dest.exists():
            if not args.force:
                raise FileExistsError(
                    f"{dest} esiste già. Usa --force per sovrascrivere."
                )
            shutil.rmtree(dest)

    # --------------------------------------
    # 2) Download + unzip con Kaggle CLI
    # --------------------------------------
    # --unzip: estrae direttamente il contenuto zip
    # -p tmp_dir: specifica la cartella di destinazione per download/estrazione
    #
    # Nota: questo comando richiede l'autenticazione Kaggle.
    run(["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(tmp_dir), "--unzip"])

    # --------------------------------------
    # 3) Trova le cartelle train/val anche se la struttura estratta ha un livello extra
    # --------------------------------------
    train_src, val_src = find_train_val(tmp_dir)

    # --------------------------------------
    # 4) Copia train e val nella struttura standard del progetto
    # --------------------------------------
    # Copiamo (non spostiamo) perché:
    # - è più robusto
    # - e poi comunque cancelliamo tmp_dir
    shutil.copytree(train_src, data_dir / "train")
    shutil.copytree(val_src, data_dir / "val")

    # --------------------------------------
    # 5) Controllo della struttura attesa (healthy/unhealthy)
    # --------------------------------------
    expected = [
        data_dir / "train" / "healthy",
        data_dir / "train" / "unhealthy",
        data_dir / "val" / "healthy",
        data_dir / "val" / "unhealthy",
    ]

    missing = [p for p in expected if not p.exists()]

    if missing:
        # Se mancano cartelle: probabilmente classi nominate in modo diverso
        print("\nATTENZIONE: struttura non perfetta, mancano queste cartelle:")
        for p in missing:
            print(" -", p)
        print("Apri data/ e verifica come sono chiamate le classi (potrebbero essere nomi diversi).")
    else:
        # Se la struttura è ok, contiamo i wav per classe.
        # Serve anche per evidenziare sbilanciamento tra classi.
        def count_wav(p: Path) -> int:
            return len(list(p.glob("*.wav")))

        print("\nOK! Dataset pronto in:", data_dir)
        print("train/healthy:", count_wav(data_dir / "train" / "healthy"))
        print("train/unhealthy:", count_wav(data_dir / "train" / "unhealthy"))
        print("val/healthy:", count_wav(data_dir / "val" / "healthy"))
        print("val/unhealthy:", count_wav(data_dir / "val" / "unhealthy"))

    # --------------------------------------
    # 6) Pulizia: rimuove la cartella temporanea del download
    # --------------------------------------
    # ignore_errors=True: evita crash se tmp_dir non esiste o è già stata rimossa
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()