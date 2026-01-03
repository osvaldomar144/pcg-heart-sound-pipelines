from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# Utility per scoprire i file e separare train/val secondo la struttura cartelle
from src.data import discover_dataset, split_items

# Funzioni di preprocessing audio (quelle che usiamo nelle pipeline)
from src.audio_ops import (
    load_audio,
    normalize_rms,
    chunk_or_pad,
    butter_bandpass,
    fft_denoise_spectral_gate,
    wavelet_denoise,
)

# Costruisce la pipeline completa (che produce lo spettrogramma Mel)
from src.pipelines import build_pipeline


# -----------------------------
# FUNZIONI DI SUPPORTO (metriche semplici)
# -----------------------------
def rms(x: np.ndarray) -> float:
    """
    RMS (Root Mean Square): misura dell'energia media del segnale.
    Serve per verificare se la normalizzazione del volume (normalize_rms)
    sta facendo il suo lavoro: RMS dopo preprocess dovrebbe essere simile tra file.
    """
    return float(np.sqrt(np.mean(x**2) + 1e-12))


def peak(x: np.ndarray) -> float:
    """
    Peak amplitude: massimo valore assoluto del segnale.
    Serve a controllare se qualche operazione (es. filtro band-pass)
    aumenta troppo i picchi (possibile instabilità o "ringing").
    """
    return float(np.max(np.abs(x)) + 1e-12)


def spectrum_mag(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola lo spettro in ampiezza tramite FFT (rFFT).
    NOTA: qui lo usiamo solo per 'audit' e visualizzazione, non per training.

    Output:
    - freqs: asse delle frequenze (Hz)
    - mag: modulo dello spettro |X(f)|
    """
    X = np.fft.rfft(x)  # FFT per segnali reali (ritorna solo frequenze >= 0)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return freqs, mag


def save_fig(path: str):
    """
    Salva una figura matplotlib in modo sicuro:
    - crea le cartelle se non esistono
    - salva in PNG ad alta risoluzione
    - chiude la figura per non consumare memoria
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# MAIN: AUDIT DEL PREPROCESSING
# -----------------------------
def main():
    """
    Scopo dello script:
    - Non allena modelli.
    - Serve per verificare che ogni pipeline di preprocessing:
        1) produce segnali di lunghezza costante
        2) non produce NaN/Inf
        3) normalizza correttamente l'energia (RMS)
        4) produce spettrogrammi Mel con shape costante e valori finiti
        5) mostra visivamente gli effetti (waveform, FFT, Mel)

    Output:
    - runs/audit/<pipeline>/sample_*.png : figure di confronto
    - runs/audit/<pipeline>/stats.json   : statistiche per file
    - runs/audit/summary.json           : riepilogo per pipeline
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Cartella dataset (contiene train/ e val/)")
    ap.add_argument("--cfg", default="configs/pipelines.json", help="Config con definizione pipeline")
    ap.add_argument("--out_dir", default="runs/audit", help="Dove salvare i risultati dell'audit")
    ap.add_argument("--n_per_class", type=int, default=3, help="Quanti esempi per classe controllare")
    args = ap.parse_args()

    # Carichiamo la configurazione JSON delle pipeline (parametri audio e TF)
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Scopriamo i file audio dal dataset e prendiamo lo split train
    items = discover_dataset(args.data_dir)
    train, _ = split_items(items)

    # Selezioniamo pochi esempi per classe (audit veloce e ripetibile)
    healthy = [x for x in train if x.label == 0][:args.n_per_class]
    unhealthy = [x for x in train if x.label == 1][:args.n_per_class]
    samples = healthy + unhealthy  # totale = 2 * n_per_class

    # Parametri globali di audio: sampling target e durata target
    sr_tgt = int(cfg["audio"]["target_sr"])
    chunk_seconds = float(cfg["audio"]["chunk_seconds"])
    chunk_len = int(sr_tgt * chunk_seconds)  # numero campioni finale atteso

    # Lista pipeline definite nel config
    pipelines = list(cfg["pipelines"].keys())
    summary = {}  # riepilogo finale per pipeline

    # Per ogni pipeline: generiamo plot + stats sugli stessi sample
    for pname in pipelines:
        # spec_fn è la pipeline COMPLETA (legge file e produce S = Mel/log/pcen ecc.)
        spec_fn = build_pipeline(cfg, pname)

        # cartella di output per questa pipeline
        pdir = os.path.join(args.out_dir, pname)
        os.makedirs(pdir, exist_ok=True)

        # stats aggregate per pipeline
        stats = {
            "files": [],  # lista di record per ogni file controllato
            "checks": {
                "target_sr": sr_tgt,
                "chunk_len_samples": chunk_len,
            },
        }

        # Eseguiamo l'audit su ciascun sample scelto
        for i, it in enumerate(samples):
            # Carichiamo il segnale grezzo, senza forzare il sampling rate
            y_raw, sr_raw = load_audio(it.path, sr=None, mono=True)

            # -------------------------
            # 1) REPLICHIAMO PREPROCESSING BASE per controlli espliciti
            # -------------------------
            # Resampling esplicito verso sr_tgt (coerenza tra file)
            if sr_raw != sr_tgt:
                import librosa
                y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=sr_tgt).astype(np.float32)
            else:
                y = y_raw.astype(np.float32)

            # Salviamo un "prima" per visualizzare l'effetto del filtro/denoise
            y_before = y.copy()

            # Applichiamo un filtro/denoise coerente con il nome pipeline.
            # Nota: questa è una "scorciatoia" per mostrare a video l'effetto del core step.
            # La pipeline completa vera e propria (spec_fn) viene comunque usata dopo.
            if "bandpass" in pname:
                bp = cfg["pipelines"][pname].get("bandpass", {"lowcut": 20, "highcut": 800, "order": 4})
                y = butter_bandpass(y, sr_tgt, bp["lowcut"], bp["highcut"], bp.get("order", 4))

            if "fft_denoise" in pname:
                dd = cfg["pipelines"][pname].get("fft_denoise", {"prop_decrease": 0.8})
                y = fft_denoise_spectral_gate(
                    y,
                    sr_tgt,
                    n_fft=cfg["tf"]["n_fft"],
                    hop_length=cfg["tf"]["hop_length"],
                    prop_decrease=dd.get("prop_decrease", 0.8),
                )

            if "wavelet_denoise" in pname:
                w = cfg["pipelines"][pname].get("wavelet", {"wavelet": "db6", "level": 4, "mode": "soft"})
                y = wavelet_denoise(y, wavelet=w["wavelet"], level=w["level"], mode=w["mode"])

            # Normalizzazione e durata fissa: queste due operazioni sono sempre presenti
            y = normalize_rms(y)
            y = chunk_or_pad(y, sr=sr_tgt, seconds=chunk_seconds)

            # -------------------------
            # 2) CHECK DI CONSISTENZA (sanity checks)
            # -------------------------
            # Controllo lunghezza
            assert len(y) == chunk_len, f"Chunk length mismatch: {len(y)} != {chunk_len}"

            # Controllo numerico (no NaN/Inf)
            assert np.isfinite(y).all(), "Found NaN/Inf in audio after preprocessing"

            # -------------------------
            # 3) PIPELINE COMPLETA: produce lo spettrogramma Mel finale
            # -------------------------
            # Qui usiamo davvero build_pipeline (quindi include tutti gli step definiti nel JSON)
            S = spec_fn(it.path)

            # Controlli su spettrogramma
            assert np.isfinite(S).all(), "Found NaN/Inf in spectrogram"
            assert S.ndim == 2, "Spectrogram must be 2D (n_mels, T)"

            # Record numerico per questo file (utile per summary e debugging)
            rec = {
                "path": it.path,
                "label": int(it.label),
                "sr_raw": int(sr_raw),
                "len_raw": int(len(y_raw)),
                "rms_raw": rms(y_raw),
                "peak_raw": peak(y_raw),
                "rms_after": rms(y),
                "peak_after": peak(y),
                "len_after": int(len(y)),
                "spec_shape": list(S.shape),
            }
            stats["files"].append(rec)

            # -------------------------
            # 4) PLOT DI DIAGNOSTICA
            # -------------------------
            # Creiamo una griglia 2x3:
            # - 1° riga: waveform raw, waveform dopo resample, waveform finale preprocessata
            # - 2° riga: FFT raw, FFT processed, Mel spectrogram
            fig = plt.figure(figsize=(12, 6))

            ax1 = fig.add_subplot(2, 3, 1)
            ax1.plot(y_raw)
            ax1.set_title("Raw waveform")
            ax1.set_xlabel("samples")

            ax2 = fig.add_subplot(2, 3, 2)
            ax2.plot(y_before[: min(len(y_before), chunk_len)])
            ax2.set_title("After resample (pre filter/denoise)")
            ax2.set_xlabel("samples")

            ax3 = fig.add_subplot(2, 3, 3)
            ax3.plot(y)
            ax3.set_title("After filter/denoise + norm + chunk")
            ax3.set_xlabel("samples")

            # FFT raw vs processed (limitiamo a 0..1000Hz per vedere banda utile)
            fr, mag_r = spectrum_mag(y_raw[: min(len(y_raw), chunk_len)], sr_raw)
            fp, mag_p = spectrum_mag(y, sr_tgt)

            ax4 = fig.add_subplot(2, 3, 4)
            ax4.plot(fr, mag_r)
            ax4.set_title("FFT magnitude (raw)")
            ax4.set_xlabel("Hz")
            ax4.set_xlim(0, 1000)

            ax5 = fig.add_subplot(2, 3, 5)
            ax5.plot(fp, mag_p)
            ax5.set_title("FFT magnitude (processed)")
            ax5.set_xlabel("Hz")
            ax5.set_xlim(0, 1000)

            ax6 = fig.add_subplot(2, 3, 6)
            ax6.imshow(S, aspect="auto", origin="lower")
            ax6.set_title(f"Mel ({pname})")
            ax6.set_xlabel("frames")
            ax6.set_ylabel("mel bins")

            # Salviamo la figura per questo sample
            out_png = os.path.join(pdir, f"sample_{i}_label{it.label}.png")
            save_fig(out_png)

        # -------------------------
        # 5) SUMMARY PER PIPELINE
        # -------------------------
        # Riepilogo: numeri utili per confrontare pipeline in modo rapido
        summary[pname] = {
            "n_files_checked": len(stats["files"]),
            "example_spec_shape": stats["files"][0]["spec_shape"] if stats["files"] else None,
            "rms_raw_mean": float(np.mean([f["rms_raw"] for f in stats["files"]])),
            "rms_after_mean": float(np.mean([f["rms_after"] for f in stats["files"]])),
            "peak_after_max": float(np.max([f["peak_after"] for f in stats["files"]])),
        }

        # Salviamo stats dettagliate per pipeline
        with open(os.path.join(pdir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    # Salviamo il riepilogo generale
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Output su console
    print("AUDIT DONE. See:", args.out_dir)
    print("Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()