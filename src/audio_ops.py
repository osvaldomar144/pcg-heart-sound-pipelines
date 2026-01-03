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


def to_mono(y: np.ndarray) -> np.ndarray:
    """
    Placeholder per conversione a mono.

    Nel nostro caso librosa.load(..., mono=True) fa già il lavoro.
    La teniamo comunque per:
    - chiarezza nella pipeline ("step" esplicito)
    - futura compatibilità se decidiamo di gestire audio multicanale manualmente
    """
    return y.astype(np.float32)


def normalize_rms(y: np.ndarray, target_rms: float = 0.1, eps: float = 1e-8) -> np.ndarray:
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


def fft_denoise_spectral_gate(
    y: np.ndarray,
    sr: int,
    n_fft: int = 512,
    hop_length: int = 128,
    prop_decrease: float = 0.8,
) -> np.ndarray:
    """
    Denoising basato su STFT (quindi: finestre + FFT -> rappresentazione in frequenza nel tempo).
    È un denoising "Fourier-like" perché lavora nello spazio tempo-frequenza.

    Idea:
    1) calcola STFT -> ottiene magnitudine (mag) e fase
    2) stima un profilo di rumore guardando i frame con energia media più bassa (lowest 10%)
    3) costruisce una mask: dove mag < noise_profile attenua
    4) ricostruisce il segnale con iSTFT mantenendo la fase originale

    Parametri:
    - n_fft, hop_length: risoluzione della STFT
    - prop_decrease: quanto attenuare sotto la soglia di rumore
      (0.8 significa "riduci parecchio", ma non azzeri completamente)

    Pro:
    - adattivo: si adatta al rumore del singolo file
    - spesso migliora robustezza su registrazioni rumorose

    Contro:
    - se la stima del rumore è sbagliata può attenuare parti utili
    - più costoso computazionalmente del semplice band-pass
    """
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.exp(1j * np.angle(stft))

    # Energia media per frame (una proxy della "loudness" frame-by-frame)
    frame_energy = mag.mean(axis=0)

    # Stimiamo il rumore usando i frame più silenziosi (10% più bassi)
    k = max(1, int(0.1 * len(frame_energy)))
    noise_frames = np.argsort(frame_energy)[:k]
    noise_profile = np.median(mag[:, noise_frames], axis=1, keepdims=True)

    # Mask: se una componente spettrale è sotto il profilo rumore -> attenua
    mask = mag >= noise_profile
    mag_d = mag * (mask + (~mask) * (1 - prop_decrease))

    # Ricostruzione nel dominio del tempo (usa la fase originale)
    y_out = librosa.istft(mag_d * phase, hop_length=hop_length, length=len(y))
    return y_out.astype(np.float32)


def wavelet_denoise(y: np.ndarray, wavelet: str = "db6", level: int = 4, mode: str = "soft") -> np.ndarray:
    """
    Denoising con Wavelet:
    - Decompone il segnale in coefficienti (approssimazione + dettagli a varie scale)
    - Stima il rumore dai coefficienti di dettaglio ad alta frequenza
    - Applica thresholding (soft/hard) ai dettagli
    - Ricostruisce il segnale con wavelet inverse

    Parametri:
    - wavelet: tipo di wavelet (db6 è comune per segnali bio/PCG)
    - level: profondità decomposizione
    - mode: "soft" o "hard" thresholding
      * soft: più “morbido”, riduce artefatti
      * hard: taglia secco sopra/sotto soglia

    Threshold:
    - U-threshold (universal threshold): sigma * sqrt(2 log N)
    - sigma stimata con MAD (median absolute deviation) / 0.6745

    Pro:
    - Buono su segnali non stazionari e rumore impulsivo
    - Spesso preserva meglio transienti rispetto a filtri lineari

    Contro:
    - Parametri (wavelet/level) influenzano molto il risultato
    - Se troppo aggressivo può “lisciare” il battito e perdere dettagli utili
    """
    coeffs = pywt.wavedec(y, wavelet, level=level)

    # coeffs[-1] sono i dettagli al livello più alto (spesso dominati dal rumore)
    detail = coeffs[-1]

    # stima sigma con MAD (robusta)
    sigma = (np.median(np.abs(detail)) / 0.6745) if detail.size else 0.0

    # universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(y) + 1))

    # Applica thresholding ai dettagli, lascia invariata l'approssimazione coeffs[0]
    coeffs_t = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode=mode) for c in coeffs[1:]]

    # Ricostruzione
    y_rec = pywt.waverec(coeffs_t, wavelet)[: len(y)]
    return y_rec.astype(np.float32)