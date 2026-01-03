from __future__ import annotations

import numpy as np
import librosa


def mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Calcola il Mel-spectrogramma (rappresentazione tempo-frequenza) a partire dal segnale audio.

    Perché lo usiamo:
    - La CNN lavora bene su input 2D tipo "immagine".
    - Il Mel-spectrogramma rende visibili pattern energetici nel tempo e nelle bande di frequenza.
    - La scala Mel è un re-mapping delle frequenze che concentra più risoluzione nelle basse frequenze
      (molto rilevanti per suoni cardiaci) e meno nelle alte.

    Parametri principali:
    - sr: sampling rate del segnale (deve essere coerente tra tutti i file)
    - n_fft: dimensione della FFT per ciascuna finestra (risoluzione in frequenza)
    - hop_length: passo tra finestre successive (risoluzione nel tempo)
    - win_length: lunghezza della finestra (spesso = n_fft)
    - window: tipo di finestra (es. 'hann') per ridurre leakage spettrale
    - n_mels: numero di bande Mel (altezza dell'immagine finale)
    - fmin, fmax: intervallo di frequenze considerato (in Hz).
      Qui è utile limitarlo alla banda “interessante” del dominio (es. 20–800 Hz per PCG).

    power=2.0:
    - restituisce lo spettro di potenza (|STFT|^2), non solo ampiezza.
      È una scelta comune per spettrogrammi usati in ML.

    Output:
    - S: matrice 2D float32 di shape (n_mels, T)
      dove T dipende da hop_length e dalla durata del segnale.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return S.astype(np.float32)


def log1p(S: np.ndarray) -> np.ndarray:
    """
    Applica la compressione logaritmica: log(1 + S).

    Perché serve:
    - Gli spettrogrammi di potenza hanno una dinamica enorme:
      poche zone con energia molto alta e tante zone con energia bassa.
    - Il log comprime la dinamica e rende più "lineare" l'informazione per la rete,
      migliorando stabilità del training e leggibilità dei pattern.

    Perché log1p (e non log):
    - log1p(x) = log(1+x) evita problemi quando x=0 (log(0) è -inf),
      quindi è numericamente più stabile.

    Output:
    - stessa shape di S, float32
    """
    return np.log1p(S).astype(np.float32)