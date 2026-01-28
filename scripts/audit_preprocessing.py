from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import librosa

# Utility per scoprire i file e separare train/val secondo la struttura cartelle
from src.data import discover_dataset, split_items

# Funzioni di preprocessing audio (quelle che usiamo nelle pipeline)
from src.audio_ops import (
    load_audio,
    to_mono,
    normalize_rms,
    normalize_peak,
    soft_clip,
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


def run_audio_steps(
    path: str,
    cfg: dict,
    pname: str,
) -> tuple[np.ndarray, int]:
    """
    Esegue gli step audio della pipeline (stessi operatori e stesso ordine).
    Si ferma prima delle trasformate (mel/log) e ritorna (y, sr).
    """
    audio_cfg = cfg["audio"]
    tf_cfg = cfg["tf"]
    pipe_cfg = cfg["pipelines"][pname]
    steps = pipe_cfg["steps"]

    y = None
    sr = None
    for step in steps:
        if step == "load":
            y, sr = load_audio(path, sr=None, mono=audio_cfg.get("mono", True))
        elif step == "resample":
            target_sr = int(audio_cfg["target_sr"])
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)
                sr = target_sr
        elif step == "to_mono":
            y = to_mono(y)
        elif step == "bandpass":
            bp = pipe_cfg.get("bandpass", {"lowcut": 20, "highcut": 800, "order": 4})
            y = butter_bandpass(
                y,
                sr=sr,
                lowcut=bp["lowcut"],
                highcut=bp["highcut"],
                order=bp.get("order", 4),
            )
        elif step == "fft_denoise":
            dd = pipe_cfg.get("fft_denoise", {"prop_decrease": 0.8})
            y = fft_denoise_spectral_gate(
                y,
                sr=sr,
                n_fft=tf_cfg["n_fft"],
                hop_length=tf_cfg["hop_length"],
                prop_decrease=dd.get("prop_decrease", 0.8),
            )
        elif step == "wavelet_denoise":
            w = pipe_cfg.get("wavelet", {"wavelet": "db6", "level": 4, "mode": "soft"})
            y = wavelet_denoise(
                y,
                wavelet=w["wavelet"],
                level=w["level"],
                mode=w["mode"],
            )
        elif step == "normalize_rms":
            y = normalize_rms(y)
        elif step == "normalize_peak":
            y = normalize_peak(y)
        elif step == "soft_clip":
            sc = pipe_cfg.get("soft_clip", {"drive": 1.5, "target_peak": 0.99})
            y = soft_clip(
                y,
                drive=sc.get("drive", 1.5),
                target_peak=sc.get("target_peak", 0.99),
            )
        elif step == "chunk_or_pad":
            y = chunk_or_pad(y, sr=sr, seconds=float(audio_cfg["chunk_seconds"]))
        elif step in ("mel", "log"):
            # Le trasformate non servono per le metriche audio
            continue
        else:
            raise ValueError(f"Step sconosciuto: {step}")

    if y is None or sr is None:
        raise RuntimeError("Pipeline audio incompleta: manca lo step 'load'.")
    return y, sr


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
            # 1) PREPROCESSING AUDIO: stessi step e operatori delle pipeline
            # -------------------------
            # Salviamo un "prima" per visualizzare l'effetto del filtro/denoise
            # (solo resample, coerente con l'opzionale step "resample")
            if sr_raw != sr_tgt:
                y_before = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=sr_tgt).astype(np.float32)
            else:
                y_before = y_raw.astype(np.float32)

            # Eseguiamo tutti gli step audio definiti dalla pipeline
            y, sr = run_audio_steps(it.path, cfg, pname)

            # -------------------------
            # 2) CHECK DI CONSISTENZA (sanity checks)
            # -------------------------
            # Controllo lunghezza
            expected_len = int(sr * chunk_seconds)
            assert len(y) == expected_len, f"Chunk length mismatch: {len(y)} != {expected_len}"

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
            ax2.plot(y_before[: min(len(y_before), len(y))])
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
