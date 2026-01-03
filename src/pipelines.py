from __future__ import annotations

from typing import Dict, Any, Callable

import numpy as np
import librosa

# Importiamo gli "operatori" di preprocessing audio (dominio tempo)
from .audio_ops import (
    load_audio,
    to_mono,
    normalize_rms,
    chunk_or_pad,
    butter_bandpass,
    fft_denoise_spectral_gate,
    wavelet_denoise,
)

# Importiamo gli "operatori" di trasformata (dominio tempo-frequenza)
from .tf_ops import mel_spectrogram, log1p


def build_pipeline(cfg: Dict[str, Any], name: str) -> Callable[[str], np.ndarray]:
    """
    Costruisce una pipeline di preprocessing + trasformata a partire da un file di config (JSON).

    Idea generale:
    - Nel config pipelines.json definiamo pipeline come sequenze di step (stringhe).
    - build_pipeline legge la lista di step e crea una funzione 'run(path) -> spectrogramma'.

    Perché è utile:
    - Possiamo definire e confrontare molte pipeline senza cambiare il codice.
    - Basta aggiungere una pipeline nel JSON e (se gli step esistono) funziona.

    Parametri:
    - cfg: dizionario caricato da configs/pipelines.json
    - name: nome della pipeline (es. "bandpass_mel", "fft_denoise_mel"...)

    Ritorna:
    - run(path): funzione che prende un percorso .wav e restituisce uno spettrogramma 2D (n_mels, T).
    """

    # Sezioni della config:
    # - audio_cfg: parametri globali audio (target_sr, chunk_seconds, mono, ...)
    # - tf_cfg: parametri globali trasformata (n_fft, hop_length, n_mels, fmin/fmax, ...)
    # - pipe_cfg: parametri specifici della pipeline (es. bandpass low/high, wavelet type...)
    audio_cfg = cfg["audio"]
    tf_cfg = cfg["tf"]
    pipe_cfg = cfg["pipelines"][name]

    # steps è una lista ordinata di stringhe.
    # L'ordine è fondamentale: ad esempio, la Mel richiede che 'y' esista e 'sr' sia quello corretto.
    steps = pipe_cfg["steps"]

    def run(path: str) -> np.ndarray:
        """
        Esegue la pipeline su un file audio.

        Variabili principali:
        - y: segnale audio nel dominio tempo (1D)
        - sr: sampling rate corrente associato a y
        - S: rappresentazione tempo-frequenza (Mel spectrogram) 2D

        Nota:
        - y e sr vengono creati da "load" e poi trasformati dagli step successivi.
        - S viene creato da "mel" e poi eventualmente trasformato (es. log).
        """

        y = None  # segnale audio (np.ndarray 1D)
        sr = None  # sample rate attuale
        S = None  # spettrogramma 2D (n_mels, T)

        # Eseguiamo gli step in ordine definito dal JSON
        for step in steps:
            # ------------------------
            # STEP: load
            # ------------------------
            if step == "load":
                # Carichiamo l'audio dal file.
                # sr=None -> manteniamo sample rate originale (poi resample esplicito)
                # mono=True -> forza mono già in fase di load
                y, sr0 = load_audio(path, sr=None, mono=audio_cfg.get("mono", True))
                sr = sr0

            # ------------------------
            # STEP: resample
            # ------------------------
            elif step == "resample":
                # Uniformiamo il sample rate (tutti i file devono avere sr_tgt).
                # Se non lo facciamo, a parità di n_fft e hop_length, le frequenze rappresentate cambiano.
                target_sr = int(audio_cfg["target_sr"])
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)
                    sr = target_sr

            # ------------------------
            # STEP: to_mono
            # ------------------------
            elif step == "to_mono":
                # In pratica è un placeholder (librosa già fa mono=True), ma mantiene la pipeline esplicita.
                y = to_mono(y)

            # ------------------------
            # STEP: bandpass
            # ------------------------
            elif step == "bandpass":
                # Filtro passa-banda (lowcut-highcut).
                # Serve a tenere solo la banda utile del suono cardiaco e ridurre rumore fuori banda.
                bp = pipe_cfg.get("bandpass", {"lowcut": 20, "highcut": 800, "order": 4})
                y = butter_bandpass(
                    y,
                    sr=sr,
                    lowcut=bp["lowcut"],
                    highcut=bp["highcut"],
                    order=bp.get("order", 4),
                )

            # ------------------------
            # STEP: fft_denoise
            # ------------------------
            elif step == "fft_denoise":
                # Denoising basato su STFT (tempo-frequenza).
                # È "Fourier-like" perché usa finestre+FFT per stimare e attenuare rumore.
                dd = pipe_cfg.get("fft_denoise", {"prop_decrease": 0.8})
                y = fft_denoise_spectral_gate(
                    y,
                    sr=sr,
                    n_fft=tf_cfg["n_fft"],
                    hop_length=tf_cfg["hop_length"],
                    prop_decrease=dd.get("prop_decrease", 0.8),
                )

            # ------------------------
            # STEP: wavelet_denoise
            # ------------------------
            elif step == "wavelet_denoise":
                # Denoising con wavelet thresholding.
                # Buono su segnali non stazionari / rumore impulsivo.
                w = pipe_cfg.get("wavelet", {"wavelet": "db6", "level": 4, "mode": "soft"})
                y = wavelet_denoise(
                    y,
                    wavelet=w["wavelet"],
                    level=w["level"],
                    mode=w["mode"],
                )

            # ------------------------
            # STEP: normalize_rms
            # ------------------------
            elif step == "normalize_rms":
                # Uniforma la scala/energia del segnale per ridurre variabilità di volume.
                y = normalize_rms(y)

            # ------------------------
            # STEP: chunk_or_pad
            # ------------------------
            elif step == "chunk_or_pad":
                # Forza durata costante in secondi:
                # - crop centrale se troppo lungo
                # - padding a zeri se troppo corto
                y = chunk_or_pad(y, sr=sr, seconds=float(audio_cfg["chunk_seconds"]))

            # ------------------------
            # STEP: mel
            # ------------------------
            elif step == "mel":
                # Trasformata in Mel-spectrogram:
                # output S: matrice 2D (n_mels, T) -> "immagine" per la CNN
                S = mel_spectrogram(
                    y=y,
                    sr=sr,
                    n_fft=tf_cfg["n_fft"],
                    hop_length=tf_cfg["hop_length"],
                    win_length=tf_cfg["win_length"],
                    window=tf_cfg["window"],
                    n_mels=tf_cfg["n_mels"],
                    fmin=tf_cfg["fmin"],
                    fmax=tf_cfg["fmax"],
                )

            # ------------------------
            # STEP: log
            # ------------------------
            elif step == "log":
                # Compressione dinamica: log(1+S)
                # Serve a ridurre differenze di scala (valori molto grandi vs piccoli),
                # rendendo più stabile la rappresentazione per la CNN.
                S = log1p(S)

            # ------------------------
            # STEP non riconosciuto
            # ------------------------
            else:
                raise ValueError(f"Step sconosciuto: {step}")

        # Alla fine ci aspettiamo che almeno lo step "mel" sia stato eseguito.
        # Se S è None significa che la pipeline è mal definita (manca "mel")
        if S is None:
            raise RuntimeError("Pipeline non ha prodotto uno spettrogramma.")

        return S

    # build_pipeline ritorna una funzione pronta all'uso: run(path) -> np.ndarray (spettrogramma)
    return run