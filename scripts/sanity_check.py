from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from src.data import discover_dataset, split_items
from src.pipelines import build_pipeline


def main():
    """
    Scopo dello script (sanity check):
    - Verifica che il dataset venga letto correttamente (train/val non vuoti).
    - Verifica che una pipeline scelta (es. bandpass_mel) riesca a:
        1) caricare un .wav
        2) fare preprocessing + trasformata
        3) produrre uno spettrogramma 2D (n_mels, T) con shape coerente
    - Produce un PNG con due spettrogrammi (uno healthy e uno unhealthy) per controllo visivo.

    È un check "veloce" (1 file per classe), diverso dall'audit che analizza più file e più dettagli.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Cartella dataset (contiene train/ e val/)")
    ap.add_argument("--cfg", default="configs/pipelines.json", help="Config con pipeline e parametri")
    ap.add_argument("--pipeline", default="bandpass_mel", help="Nome pipeline da testare")
    ap.add_argument("--out_png", default="runs/sanity.png", help="Dove salvare l'immagine di controllo")
    args = ap.parse_args()

    # Carichiamo la configurazione JSON (contiene definizione pipeline e parametri audio/TF)
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Scopriamo tutti i file wav e separiamo train e val
    items = discover_dataset(args.data_dir)
    train, val = split_items(items)

    # Stampa utile: ci conferma che il dataset è stato trovato e la struttura è corretta
    print(f"Train: {len(train)}  Val: {len(val)}")

    # Costruiamo la funzione pipeline che, data la path di un wav, ritorna uno spettrogramma 2D
    spec_fn = build_pipeline(cfg, args.pipeline)

    # Selezioniamo un esempio healthy e uno unhealthy dal TRAIN
    # next(...) prende il primo elemento che soddisfa la condizione
    h = next(x for x in train if x.label == 0)  # healthy
    u = next(x for x in train if x.label == 1)  # unhealthy

    # Applichiamo la pipeline ai due file
    # Output atteso: array 2D con shape (n_mels, T)
    Sh = spec_fn(h.path)
    Su = spec_fn(u.path)

    # Stampiamo shape: serve per verificare che:
    # - la pipeline produce output coerente
    # - la chunk length e parametri TF generano sempre la stessa dimensione
    print("Spectrogram shapes:", Sh.shape, Su.shape, " (n_mels, T)")

    # Creiamo una figura con due pannelli:
    # - a sinistra spettrogramma healthy
    # - a destra spettrogramma unhealthy
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(Sh, aspect="auto", origin="lower")
    ax1.set_title(f"Healthy - {args.pipeline}")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Mel bins")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(Su, aspect="auto", origin="lower")
    ax2.set_title(f"Unhealthy - {args.pipeline}")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Mel bins")

    # Salviamo il PNG nella cartella desiderata
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)

    print("Saved:", args.out_png)


if __name__ == "__main__":
    main()