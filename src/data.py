from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

# Mappatura testuale -> numerica delle classi.
# Convenzione:
# 0 = healthy
# 1 = unhealthy
#
# Usare interi è comodo perché:
# - i modelli ML/DL lavorano con label numeriche
# - CrossEntropyLoss in PyTorch si aspetta classi 0..C-1
LABELS = {"healthy": 0, "unhealthy": 1}


@dataclass(frozen=True)
class FileItem:
    """
    Struttura dati "leggera" che rappresenta un singolo file audio nel dataset.

    campi:
    - path: percorso del file .wav
    - label: etichetta numerica (0=healthy, 1=unhealthy)
    - split: da quale split proviene ("train" o "val")

    frozen=True:
    - rende l'oggetto immutabile (più sicuro: nessuno cambia path/label per errore a runtime)
    - utile quando si passa in giro la lista di file tra funzioni
    """
    path: str
    label: int
    split: str  # train / val


def discover_dataset(data_dir: str) -> List[FileItem]:
    """
    Scansiona la cartella dataset e costruisce una lista di FileItem.

    Struttura attesa:
      data_dir/
        train/
          healthy/*.wav
          unhealthy/*.wav
        val/
          healthy/*.wav
          unhealthy/*.wav

    Perché implementato così:
    - Evita hardcoding di liste file: basta mettere i wav nelle cartelle giuste.
    - Associa automaticamente label e split usando la struttura del filesystem.

    Ritorna:
    - lista di FileItem (train+val tutti insieme), ognuno con path/label/split.

    Eccezioni:
    - Se non trova nessun .wav, lancia FileNotFoundError: tipicamente dataset non scaricato
      o percorso data_dir sbagliato.
    """
    items: List[FileItem] = []

    # Cicliamo sui due split (train e val)
    for split in ["train", "val"]:
        # Cicliamo sulle due classi
        for cls in ["healthy", "unhealthy"]:
            folder = Path(data_dir) / split / cls

            # Se la cartella non esiste, saltiamo.
            # (così lo script non crasha subito se manca una cartella, ma alla fine controlliamo items)
            if not folder.exists():
                continue

            # Per ogni .wav dentro la cartella, creiamo un FileItem
            for p in folder.glob("*.wav"):
                items.append(FileItem(path=str(p), label=LABELS[cls], split=split))

    # Se la lista è vuota, molto probabilmente:
    # - dataset non è stato scaricato
    # - cartelle non sono come attese
    # - data_dir è sbagliato
    if not items:
        raise FileNotFoundError(
            f"Nessun .wav trovato. Struttura attesa: {data_dir}/train/(healthy|unhealthy)/*.wav e val/..."
        )

    return items


def split_items(items: List[FileItem]) -> Tuple[List[FileItem], List[FileItem]]:
    """
    Separa la lista completa in (train_items, val_items) usando il campo 'split'.

    Perché serve:
    - Alcuni script (train_cnn, train_baseline, audit, sanity_check) vogliono train e val separati.
    - Tenere discover_dataset separato da split_items rende il codice più modulare.

    Ritorna:
    - (train, val) come due liste di FileItem.

    Eccezioni:
    - Se una delle due liste è vuota, lancia ValueError:
      probabilmente manca train/ o val/ nel dataset.
    """
    train = [x for x in items if x.split == "train"]
    val = [x for x in items if x.split == "val"]

    if not train or not val:
        raise ValueError("Manca train o val. Controlla le cartelle.")

    return train, val