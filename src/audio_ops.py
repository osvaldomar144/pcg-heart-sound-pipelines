from __future__ import annotations

import numpy as np
import librosa
import scipy.signal as sps
import pywt


def load_audio(path: str, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Carica un file audio da disco.

    Parametri:
    - path: percorso del file .wav
    - sr: se None, mantiene il sample rate originale del file.
          se un intero, librosa fa resampling automaticamente a quel sr.
    - mono: se True, converte in mono (media dei canali) durante il load.

    Ritorna:
    - y: array numpy float32 con i campioni audio
    - sample_rate: sample rate usato (se sr è specificato, coincide con sr; altrimenti quello del file)
    """
    y, file_sr = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), (sr if sr is not None else file_sr)


def normalize_rms(y: np.ndarray, target_rms: float = 0.05, eps: float = 1e-8) -> np.ndarray:
    """
    Normalizzazione del volume basata su RMS (energia media).

    Perché serve:
    - Registrazioni diverse possono avere volumi molto diversi.
    - Senza normalizzazione il modello potrebbe imparare "volume = classe", cioè un bias.
    - Uniformare RMS rende più confrontabili i campioni.

    Come funziona:
    - calcola RMS(y)
    - scala il segnale in modo che RMS diventi target_rms

    target_rms:
    - 0.1 è un valore “moderato”: evita segnali troppo piccoli ma non spara tutto al massimo.

    Nota:
    - Questa normalizzazione NON limita i picchi massimi (peak).
      Se un filtro introduce overshoot, i picchi possono crescere anche se RMS resta controllato.
    """
    rms = np.sqrt(np.mean(y**2) + eps)
    return (y / (rms + eps) * target_rms).astype(np.float32)


def normalize_peak(y: np.ndarray, target_peak: float = 0.99, eps: float = 1e-8) -> np.ndarray:
    """
    Normalizzazione per ampiezza massima (peak).

    - Calcola il valore assoluto massimo del segnale.
    - Scala il segnale in modo che il picco diventi target_peak.

    target_peak:
    - 0.99 evita il clipping quando il segnale viene salvato in formati con range [-1, 1].
    """
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= eps:
        return y.astype(np.float32)
    return (y / peak * target_peak).astype(np.float32)



def chunk_or_pad(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """
    Forza tutti i segnali ad avere la stessa durata (fondamentale per input CNN fisso).

    - Se il segnale è più lungo: fa un crop centrale (center crop)
      -> scelta semplice e stabile (non dipende dall'inizio/fine)
    - Se il segnale è più corto: fa padding con zeri in modo simmetrico

    Parametri:
    - sr: sample rate
    - seconds: durata target in secondi

    Output:
    - y con len = sr * seconds

    Nota:
    - Padding con zeri può introdurre “silenzio” artificiale.
      In audio cardiaco spesso va bene, ma alternativamente si potrebbe:
      * ripetere il segnale (loop)
      * prendere più chunk dal file (data augmentation)
    """
    target_len = int(sr * seconds)

    if len(y) == target_len:
        return y

    if len(y) > target_len:
        # Center crop: prende una finestra centrata
        start = (len(y) - target_len) // 2
        return y[start : start + target_len]

    # Pad simmetrico
    pad_total = target_len - len(y)
    left = pad_total // 2
    right = pad_total - left
    return np.pad(y, (left, right), mode="constant").astype(np.float32)


def trim(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """
    Taglia o padding del segnale usando sempre l'inizio.

    - Se il segnale è più lungo: prende i primi campioni (start crop).
    - Se il segnale è più corto: pad con zeri alla fine.
    """
    target_len = int(sr * seconds)

    if len(y) == target_len:
        return y

    if len(y) > target_len:
        return y[:target_len]

    pad_total = target_len - len(y)
    return np.pad(y, (0, pad_total), mode="constant").astype(np.float32)


def butter_bandpass(y: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """
    Filtro passa-banda Butterworth applicato con filtfilt (zero-phase).

    Perché serve:
    - Nei suoni cardiaci una grossa parte dell’informazione utile sta in una banda limitata.
    - Rumori a bassissima frequenza (movimento) o alta frequenza (fruscio) possono disturbare.

    Dettagli tecnici:
    - Butterworth: risposta in frequenza "liscia" (senza ripple).
    - filtfilt: applica il filtro avanti e indietro -> niente fase (non sposta gli eventi nel tempo).

    Attenzione:
    - filtfilt può introdurre overshoot/ringing e aumentare picchi locali.
      È esattamente ciò che si è visto nell’audit (peak_after_max alto).
      Per questo, in alcune pipeline si possono aggiungere un limiter/soft-clip dopo.
    """
    nyq = 0.5 * sr  # Nyquist frequency
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.99999)

    b, a = sps.butter(order, [low, high], btype="band")
    return sps.filtfilt(b, a, y).astype(np.float32)


def soft_clip(y: np.ndarray, drive: float = 1.5, target_peak: float = 0.99, eps: float = 1e-8) -> np.ndarray:
    """
    Soft clipping per ridurre i picchi locali (es. dopo filtfilt).

    - Applica una non-linearità morbida (tanh) per comprimere i picchi.
    - Poi riporta l'ampiezza massima a target_peak.

    Parametri:
    - drive: quanto spingere il segnale prima del clipping (più alto = più compressione).
    - target_peak: picco massimo desiderato dopo il soft-clip.
    """
    if not y.size:
        return y.astype(np.float32)

    y_drive = y * float(drive)
    y_soft = np.tanh(y_drive)

    peak = float(np.max(np.abs(y_soft)))
    if peak <= eps:
        return y_soft.astype(np.float32)
    return (y_soft / peak * target_peak).astype(np.float32)


def stft(
    y: np.ndarray,
    n_fft: int = 512,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """
    Calcola la STFT (Short-Time Fourier Transform) del segnale in ingresso.

    Parametri:
    - y: segnale 1D
    - n_fft: dimensione FFT
    - hop_length: passo tra finestre (default: n_fft // 4)
    - win_length: lunghezza finestra (default: n_fft)
    - window: tipo di finestra per librosa
    - center: se True, centra i frame (padding riflesso)

    Ritorna:
    - matrice reale (freq x frame) con la magnitudine della STFT
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    stft_complex = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )
    return np.abs(stft_complex).astype(np.float32)


def wavelet_transform(
    y: np.ndarray,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
    sr: int | None = None,
    scale_min: float = 1.0,
    scale_max: float | None = 128.0,
    num_scales: int = 64,
    scale_spacing: str = "log",
    output: str = "magnitude",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Trasformata wavelet continua (CWT) del segnale in ingresso.

    Parametri:
    - y: segnale 1D
    - wavelet: nome wavelet per pywt (es. "morl", "mexh", "cmor")
    - scales: array di scale; se None viene generato automaticamente
    - sr: sample rate; se fornito, ritorna anche le frequenze corrispondenti alle scale
    - scale_min/scale_max/num_scales/scale_spacing: usati per generare le scale
      * scale_spacing: "log" o "linear"
    - output: "magnitude" (default) o "complex"

    Ritorna:
    - coeffs: matrice (n_scales x n_samples) della CWT
    - freqs: array di frequenze (Hz) se sr è fornito, altrimenti None
    """
    if scales is None:
        max_scale = float(scale_max) if scale_max is not None else 128.0
        if scale_spacing == "linear":
            scales = np.linspace(scale_min, max_scale, num_scales, dtype=np.float32)
        else:
            scales = np.logspace(np.log10(scale_min), np.log10(max_scale), num_scales).astype(np.float32)

    coeffs, _ = pywt.cwt(y, scales, wavelet)

    if output == "magnitude":
        coeffs = np.abs(coeffs)

    freqs = None
    if sr is not None:
        freqs = pywt.scale2frequency(wavelet, scales) * float(sr)

    return coeffs.astype(np.float32), (None if freqs is None else freqs.astype(np.float32))
