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
    stft,
    fft_denoise_spectral_gate,
    wavelet_denoise,
    wavelet_transform,
)
from src.tf_ops import mel_spectrogram, log1p


def spectrum_mag(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.fft.rfft(x)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return freqs, mag


def step_params(step: str, cfg: dict, pipe_cfg: dict) -> dict:
    audio_cfg = cfg["audio"]
    tf_cfg = cfg["tf"]

    if step == "load":
        return {"sr": "originale", "mono": audio_cfg.get("mono", True)}
    if step == "resample":
        return {"target_sr": int(audio_cfg["target_sr"])}
    if step == "to_mono":
        return {}
    if step == "bandpass":
        bp = pipe_cfg.get("bandpass", {"lowcut": 20, "highcut": 800, "order": 4})
        return {
            "lowcut": bp["lowcut"],
            "highcut": bp["highcut"],
            "order": bp.get("order", 4),
        }
    if step == "fft_denoise":
        dd = pipe_cfg.get("fft_denoise", {"prop_decrease": 0.8})
        return {
            "n_fft": tf_cfg["n_fft"],
            "hop_length": tf_cfg["hop_length"],
            "prop_decrease": dd.get("prop_decrease", 0.8),
        }
    if step == "wavelet_denoise":
        w = pipe_cfg.get("wavelet", {"wavelet": "db6", "level": 4, "mode": "soft"})
        return {
            "wavelet": w["wavelet"],
            "level": w["level"],
            "mode": w["mode"],
        }
    if step == "normalize_rms":
        return {}
    if step == "normalize_peak":
        return {}
    if step == "soft_clip":
        sc = pipe_cfg.get("soft_clip", {"drive": 1.5, "target_peak": 0.99})
        return {
            "drive": sc.get("drive", 1.5),
            "target_peak": sc.get("target_peak", 0.99),
        }
    if step == "chunk_or_pad":
        return {"seconds": float(audio_cfg["chunk_seconds"])}
    if step == "trim":
        return {"seconds": float(audio_cfg["chunk_seconds"])}
    if step == "mel":
        return {
            "n_fft": tf_cfg["n_fft"],
            "hop_length": tf_cfg["hop_length"],
            "win_length": tf_cfg["win_length"],
            "window": tf_cfg["window"],
            "n_mels": tf_cfg["n_mels"],
            "fmin": tf_cfg["fmin"],
            "fmax": tf_cfg["fmax"],
        }
    if step == "stft":
        return {
            "n_fft": tf_cfg["n_fft"],
            "hop_length": tf_cfg["hop_length"],
            "win_length": tf_cfg["win_length"],
            "window": tf_cfg["window"],
        }
    if step == "wavelet_transform":
        w = pipe_cfg.get(
            "wavelet_transform",
            {
                "wavelet": "morl",
                "scale_min": 1.0,
                "scale_max": 128.0,
                "num_scales": 64,
                "scale_spacing": "log",
                "output": "magnitude",
            },
        )
        return {
            "wavelet": w.get("wavelet", "morl"),
            "scale_min": float(w.get("scale_min", 1.0)),
            "scale_max": float(w.get("scale_max", 128.0)),
            "num_scales": int(w.get("num_scales", 64)),
            "scale_spacing": w.get("scale_spacing", "log"),
            "output": w.get("output", "magnitude"),
        }
    if step == "log":
        return {"transform": "log1p"}
    return {}


def write_pipeline_summary(
    out_path: str,
    cfg: dict,
    cfg_path: str,
    pipeline_name: str,
    seed: int,
    healthy_path: str,
    unhealthy_path: str,
) -> None:
    pipe_cfg = cfg["pipelines"][pipeline_name]
    steps = pipe_cfg["steps"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"run_datetime: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"pipeline: {pipeline_name}\n")
        f.write(f"cfg_file: {cfg_path}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"sample_healthy: {healthy_path}\n")
        f.write(f"sample_unhealthy: {unhealthy_path}\n")
        f.write("\n")
        f.write("steps:\n")
        for i, step in enumerate(steps, start=1):
            params = step_params(step, cfg, pipe_cfg)
            f.write(f"{i:02d}. {step}\n")
            if params:
                for k, v in params.items():
                    f.write(f"    - {k}: {v}\n")
            else:
                f.write("    - params: none\n")


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
    spec_kind = None
    spec_freqs = None
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
            spec_kind = "spec"
            spec_freqs = None
            outputs.append({"step": step, "kind": "spec", "data": S})
        elif step == "stft":
            S = stft(
                y=y,
                n_fft=tf_cfg["n_fft"],
                hop_length=tf_cfg["hop_length"],
                win_length=tf_cfg["win_length"],
                window=tf_cfg["window"],
            )
            spec_kind = "spec"
            spec_freqs = None
            outputs.append({"step": step, "kind": "spec", "data": S})
        elif step == "wavelet_transform":
            w = pipe_cfg.get(
                "wavelet_transform",
                {
                    "wavelet": "morl",
                    "scale_min": 1.0,
                    "scale_max": 128.0,
                    "num_scales": 64,
                    "scale_spacing": "log",
                    "output": "magnitude",
                },
            )
            S, freqs = wavelet_transform(
                y=y,
                wavelet=w.get("wavelet", "morl"),
                scales=None,
                sr=sr,
                scale_min=float(w.get("scale_min", 1.0)),
                scale_max=float(w.get("scale_max", 128.0)),
                num_scales=int(w.get("num_scales", 64)),
                scale_spacing=w.get("scale_spacing", "log"),
                output=w.get("output", "magnitude"),
            )
            spec_kind = "cwt"
            spec_freqs = freqs
            outputs.append(
                {
                    "step": step,
                    "kind": "cwt",
                    "data": S,
                    "sr": sr,
                    "freqs": freqs,
                    "n_samples": len(y),
                }
            )
        elif step == "log":
            if S is None:
                raise RuntimeError("Step 'log' senza spettrogramma: manca 'mel' nella pipeline.")
            S = log1p(S)
            outputs.append(
                {
                    "step": step,
                    "kind": spec_kind or "spec",
                    "data": S,
                    "freqs": spec_freqs,
                    "sr": sr,
                    "n_samples": None if y is None else len(y),
                }
            )
        else:
            raise ValueError(f"Step sconosciuto: {step}")

    return outputs


def plot_step(
    out_path: str,
    step: str,
    a: np.ndarray,
    b: np.ndarray,
    kind: str,
    sr: int | None,
    file_a: str,
    file_b: str,
    freqs_a: np.ndarray | None,
    freqs_b: np.ndarray | None,
    n_samples_a: int | None,
    n_samples_b: int | None,
) -> None:
    name_a = os.path.basename(file_a)
    name_b = os.path.basename(file_b)
    if kind in ("spec", "cwt"):
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        if kind == "cwt" and sr is not None and freqs_a is not None and freqs_b is not None:
            t_a = (0.0, (n_samples_a or a.shape[1]) / sr)
            t_b = (0.0, (n_samples_b or b.shape[1]) / sr)
            y_a0, y_a1 = (float(freqs_a[-1]), float(freqs_a[0])) if freqs_a[0] > freqs_a[-1] else (float(freqs_a[0]), float(freqs_a[-1]))
            y_b0, y_b1 = (float(freqs_b[-1]), float(freqs_b[0])) if freqs_b[0] > freqs_b[-1] else (float(freqs_b[0]), float(freqs_b[-1]))
            ax1.imshow(
                a,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[t_a[0], t_a[1], y_a0, y_a1],
            )
            ax2.imshow(
                b,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[t_b[0], t_b[1], y_b0, y_b1],
            )
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_xlabel("Time (s)")
            ax2.set_xlabel("Time (s)")
        else:
            ax1.imshow(a, aspect="auto", origin="lower")
            ax2.imshow(b, aspect="auto", origin="lower")
            ax1.set_ylabel("Frequency bins")
            ax1.set_xlabel("Frames")
            ax2.set_xlabel("Frames")
        ax1.set_title(f"Healthy - {step} ({name_a})")
        ax2.set_title(f"Unhealthy - {step} ({name_b})")
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
        ax1.set_title(f"Healthy - {step} (waveform) ({name_a})")
        ax2.set_title(f"Unhealthy - {step} (waveform) ({name_b})")

        if sr is None:
            raise RuntimeError("Sample rate mancante per plotting FFT.")
        freqs_a, mag_a = spectrum_mag(a, sr)
        freqs_b, mag_b = spectrum_mag(b, sr)
        ax3.plot(freqs_a, mag_a)
        ax4.plot(freqs_b, mag_b)
        ax3.set_xlabel("Hz")
        ax4.set_xlabel("Hz")
        ax3.set_ylabel("Magnitude")
        ax3.set_title(f"Healthy - FFT ({name_a})")
        ax4.set_title(f"Unhealthy - FFT ({name_b})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Cartella dataset (contiene train/ e val/)")
    ap.add_argument("--cfg", default="configs/pipelines.json", help="Config con pipeline e parametri")
    ap.add_argument("--pipeline", default="bandpass_mel", help="Nome pipeline da testare")
    ap.add_argument("--seed", type=int, default=13)
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
    summary_path = os.path.join(run_dir, "pipeline_summary.txt")
    write_pipeline_summary(
        summary_path,
        cfg=cfg,
        cfg_path=args.cfg,
        pipeline_name=args.pipeline,
        seed=args.seed,
        healthy_path=h.path,
        unhealthy_path=u.path,
    )

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
            h.path,
            u.path,
            sh.get("freqs"),
            su.get("freqs"),
            sh.get("n_samples"),
            su.get("n_samples"),
        )

    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
