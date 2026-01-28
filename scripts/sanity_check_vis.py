from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import librosa

from src.data import discover_dataset, split_items
from src.audio_ops import (
    load_audio,
    to_mono,
    normalize_rms,
    normalize_peak,
    soft_clip,
    trim,
    chunk_or_pad,
    butter_bandpass,
    fft_denoise_spectral_gate,
    wavelet_denoise,
)
from src.tf_ops import mel_spectrogram, log1p


def spectrum_mag(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.rfft(x)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return freqs, mag


def run_steps(path: str, cfg: dict, pname: str) -> list[dict]:
    """
    Esegue la pipeline step-by-step e ritorna gli output intermedi (audio o spettrogramma).
    """
    audio_cfg = cfg["audio"]
    tf_cfg = cfg["tf"]
    pipe_cfg = cfg["pipelines"][pname]
    steps = pipe_cfg["steps"]

    y = None
    sr = None
    S = None
    outputs: list[dict] = []

    for step in steps:
        if step == "load":
            y, sr = load_audio(path, sr=None, mono=audio_cfg.get("mono", True))
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "resample":
            target_sr = int(audio_cfg["target_sr"])
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)
                sr = target_sr
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "to_mono":
            y = to_mono(y)
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "bandpass":
            bp = pipe_cfg.get("bandpass", {"lowcut": 20, "highcut": 800, "order": 4})
            y = butter_bandpass(
                y,
                sr=sr,
                lowcut=bp["lowcut"],
                highcut=bp["highcut"],
                order=bp.get("order", 4),
            )
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "fft_denoise":
            dd = pipe_cfg.get("fft_denoise", {"prop_decrease": 0.8})
            y = fft_denoise_spectral_gate(
                y,
                sr=sr,
                n_fft=tf_cfg["n_fft"],
                hop_length=tf_cfg["hop_length"],
                prop_decrease=dd.get("prop_decrease", 0.8),
            )
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "wavelet_denoise":
            w = pipe_cfg.get("wavelet", {"wavelet": "db6", "level": 4, "mode": "soft"})
            y = wavelet_denoise(y, wavelet=w["wavelet"], level=w["level"], mode=w["mode"])
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "normalize_rms":
            y = normalize_rms(y)
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "normalize_peak":
            y = normalize_peak(y)
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "soft_clip":
            sc = pipe_cfg.get("soft_clip", {"drive": 1.5, "target_peak": 0.99})
            y = soft_clip(
                y,
                drive=sc.get("drive", 1.5),
                target_peak=sc.get("target_peak", 0.99),
            )
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "chunk_or_pad":
            y = chunk_or_pad(y, sr=sr, seconds=float(audio_cfg["chunk_seconds"]))
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "trim":
            y = trim(y, sr=sr, seconds=float(audio_cfg["chunk_seconds"]))
            outputs.append({"step": step, "kind": "audio", "sr": sr, "data": y})
        elif step == "mel":
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
            outputs.append({"step": step, "kind": "spec", "data": S})
        elif step == "log":
            if S is None:
                raise RuntimeError("Step 'log' senza spettrogramma: manca 'mel' nella pipeline.")
            S = log1p(S)
            outputs.append({"step": step, "kind": "spec", "data": S})
        else:
            raise ValueError(f"Step sconosciuto: {step}")

    return outputs


def plot_step(out_path: str, step: str, a: np.ndarray, b: np.ndarray, kind: str, sr: int | None) -> None:
    if kind == "spec":
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.imshow(a, aspect="auto", origin="lower")
        ax2.imshow(b, aspect="auto", origin="lower")
        ax1.set_ylabel("Mel bins")
        ax1.set_xlabel("Frames")
        ax2.set_xlabel("Frames")
        ax1.set_title(f"Healthy - {step}")
        ax2.set_title(f"Unhealthy - {step}")
    else:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        ax1.plot(a)
        ax2.plot(b)
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Samples")
        ax2.set_xlabel("Samples")
        ax1.set_title(f"Healthy - {step} (waveform)")
        ax2.set_title(f"Unhealthy - {step} (waveform)")

        if sr is None:
            raise RuntimeError("Sample rate mancante per plotting FFT.")
        freqs_a, mag_a = spectrum_mag(a, sr)
        freqs_b, mag_b = spectrum_mag(b, sr)
        ax3.plot(freqs_a, mag_a)
        ax4.plot(freqs_b, mag_b)
        ax3.set_xlabel("Hz")
        ax4.set_xlabel("Hz")
        ax3.set_ylabel("Magnitude")
        ax3.set_title("Healthy - FFT")
        ax4.set_title("Unhealthy - FFT")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Cartella dataset (contiene train/ e val/)")
    ap.add_argument("--cfg", default="configs/pipelines.json", help="Config con pipeline e parametri")
    ap.add_argument("--pipeline", default="bandpass_mel", help="Nome pipeline da testare")
    ap.add_argument("--seed", type=int, default=78)
    ap.add_argument("--out_dir", default="runs/sanity", help="Cartella base output")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    items = discover_dataset(args.data_dir)
    train, _ = split_items(items)

    healthy = [x for x in train if x.label == 0]
    unhealthy = [x for x in train if x.label == 1]
    if not healthy or not unhealthy:
        raise RuntimeError("Dataset non contiene entrambe le classi (healthy/unhealthy).")

    rng = np.random.default_rng(args.seed)
    h = healthy[int(rng.integers(0, len(healthy)))]
    u = unhealthy[int(rng.integers(0, len(unhealthy)))]

    steps_h = run_steps(h.path, cfg, args.pipeline)
    steps_u = run_steps(u.path, cfg, args.pipeline)
    if [s["step"] for s in steps_h] != [s["step"] for s in steps_u]:
        raise RuntimeError("Gli step tra healthy/unhealthy non coincidono.")

    run_dir = os.path.join(
        args.out_dir,
        f"pipeline_{args.pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)

    for idx, (sh, su) in enumerate(zip(steps_h, steps_u), start=1):
        step = sh["step"]
        out_path = os.path.join(run_dir, f"{idx:02d}_{step}.png")
        plot_step(
            out_path,
            step,
            sh["data"],
            su["data"],
            sh["kind"],
            sh.get("sr"),
        )

    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
