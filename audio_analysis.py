#!/usr/bin/env python3
"""
Audio Analysis Script — Apollo 13 Flight Director Loop
=======================================================
Compares original and enhanced audio via:
  • SNR metrics (global, segmental, improvement)
  • Waveform + spectrogram plots
  • Whisper ASR transcription (before vs. after)

Usage:
  python audio_analysis.py [--input FILE] [--enhanced-dir DIR] [--start SEC] [--duration SEC]
"""

import argparse
import csv
import os
import sys
import textwrap
import time

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for plot saving
import matplotlib.pyplot as plt
import soundfile as sf
import webvtt
import jiwer


# ---------------------------------------------------------------------------
# SNR estimation (no clean reference — frame-energy VAD approach)
# ---------------------------------------------------------------------------

def estimate_snr_metrics(sig, sr=16000, frame_length=512, hop_length=256,
                         noise_pct=20, speech_pct=70):
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) < frame_length:
        p = float(np.mean(sig ** 2) + 1e-12)
        return dict(global_snr_db=np.nan, segmental_snr_db=np.nan,
                    speech_power=p, noise_power=p, speech_frames_pct=0.0)

    frames = librosa.util.frame(sig, frame_length=frame_length, hop_length=hop_length)
    frame_power = np.mean(frames ** 2, axis=0) + 1e-12
    frame_rms = np.sqrt(frame_power)

    noise_thr = np.percentile(frame_rms, noise_pct)
    speech_thr = np.percentile(frame_rms, speech_pct)

    noise_mask = frame_rms <= noise_thr
    speech_mask = frame_rms >= speech_thr

    if not np.any(noise_mask):
        noise_mask = frame_rms <= np.median(frame_rms)
    if not np.any(speech_mask):
        speech_mask = frame_rms >= np.median(frame_rms)

    noise_power = float(np.mean(frame_power[noise_mask]) + 1e-12)
    speech_power = float(np.mean(frame_power[speech_mask]) + 1e-12)
    global_snr_db = float(10.0 * np.log10(speech_power / noise_power))

    speech_frame_snr = 10.0 * np.log10(frame_power[speech_mask] / noise_power)
    speech_frame_snr = np.clip(speech_frame_snr, -10.0, 35.0)
    segmental_snr_db = float(np.mean(speech_frame_snr)) if len(speech_frame_snr) else np.nan

    return dict(
        global_snr_db=global_snr_db,
        segmental_snr_db=segmental_snr_db,
        speech_power=speech_power,
        noise_power=noise_power,
        speech_frames_pct=float(100.0 * np.mean(speech_mask)),
    )


# ---------------------------------------------------------------------------
# Plotting helpers (OOM-safe for long audio)
# ---------------------------------------------------------------------------

def plot_waveforms(signals: dict, sr: int, save_path: str,
                   max_plot_seconds: float = 120.0):
    names = list(signals.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 2.5 * len(names)),
                             sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        sig = signals[name]
        if max_plot_seconds is not None:
            max_samples = int(max_plot_seconds * sr)
            sig = sig[:max_samples]
        t = np.arange(len(sig)) / sr
        ax.plot(t, sig, linewidth=0.5)
        ax.set_title(f"{name} (first {len(sig)/sr:.1f}s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveform Comparison", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved waveform plot → {save_path}")


def _iter_chunks(total_samples, sr, chunk_seconds=12, max_plot_seconds=120):
    chunk_samples = max(int(chunk_seconds * sr), 1)
    max_samples = total_samples
    if max_plot_seconds is not None:
        max_samples = min(total_samples, int(max_plot_seconds * sr))
    start = 0
    while start < max_samples:
        end = min(start + chunk_samples, max_samples)
        yield start, end
        start = end


def plot_spectrograms(signals: dict, sr: int, save_path: str,
                      n_fft: int = 1024, hop_length: int = 256,
                      max_plot_seconds: float = 120.0):
    names = list(signals.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3 * len(names)),
                             sharex=True)
    if len(names) == 1:
        axes = [axes]

    global_mappable = None

    for ax, name in zip(axes, names):
        sig = np.asarray(signals[name], dtype=np.float32)
        peak = float(np.max(np.abs(sig)) + 1e-12)
        plotted_seconds = min(len(sig) / sr,
                             max_plot_seconds if max_plot_seconds else len(sig) / sr)

        if len(sig) < n_fft:
            ax.text(0.5, 0.5, "Signal too short", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name)
            continue

        for start, end in _iter_chunks(len(sig), sr, max_plot_seconds=max_plot_seconds):
            chunk = sig[start:end]
            if len(chunk) < n_fft:
                continue
            S = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=peak)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                           sr=sr, hop_length=hop_length)
            times = times + (start / sr)
            global_mappable = ax.pcolormesh(times, freqs, S_db,
                                           shading="auto", cmap="magma",
                                           vmin=-80, vmax=0)

        ax.set_title(f"{name} (first {plotted_seconds:.1f}s)")
        ax.set_ylabel("Hz")
        ax.grid(False)

    axes[-1].set_xlabel("Time (s)")
    if global_mappable is not None:
        fig.colorbar(global_mappable, ax=axes, format="%+2.0f dB")

    fig.suptitle("Spectrogram Comparison", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spectrogram plot → {save_path}")


# ---------------------------------------------------------------------------
# Whisper ASR
# ---------------------------------------------------------------------------

def transcribe_with_whisper(audio_path: str, model_name: str = "base"):
    """Run Whisper ASR on a single audio file. Returns text + metadata."""
    import whisper

    print(f"  Transcribing with Whisper ({model_name}): {os.path.basename(audio_path)} …")
    t0 = time.time()
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language="en")
    elapsed = time.time() - t0
    text = result.get("text", "").strip()
    word_count = len(text.split())
    print(f"    → {word_count} words, {elapsed:.1f}s")
    return {
        "text": text,
        "word_count": word_count,
        "elapsed_s": round(elapsed, 1),
        "segments": result.get("segments", []),
    }


# ---------------------------------------------------------------------------
# VTT Ground Truth
# ---------------------------------------------------------------------------

def extract_ground_truth(vtt_path: str, start_sec: float, duration: float) -> str:
    """Extract text from a VTT file for the given time window, handling YouTube rolling format."""
    if not os.path.exists(vtt_path):
        return ""
        
    end_sec = start_sec + duration
    
    def parse_time(tc: str) -> float:
        parts = tc.split(':')
        s = float(parts[-1])
        m = int(parts[-2]) if len(parts) > 1 else 0
        h = int(parts[-3]) if len(parts) > 2 else 0
        return h * 3600 + m * 60 + s
        
    captions = webvtt.read(vtt_path)
    lines = []
    
    for caption in captions:
        c_start = parse_time(caption.start)
        c_end = parse_time(caption.end)
        
        # Check overlap
        if c_start < end_sec and c_end > start_sec:
            import re
            # Remove <c> tags and timestamps like <00:00:00.000>
            clean_text = re.sub(r'<[^>]+>', '', caption.text).strip()
            
            # YouTube VTTs are "rolling" (line 2 becomes line 1 in the next cue).
            # We split by newline and only add lines that uniquely advance the transcript.
            for line in clean_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Add line only if it's not the same as the last line we added
                if not lines or line != lines[-1]:
                    lines.append(line)
                
    return " ".join(lines)

# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(input_path: str, enhanced_dir: str, output_dir: str,
                 vtt_path: str, sr: int, start: float, duration: float,
                 whisper_model: str):

    os.makedirs(output_dir, exist_ok=True)
    transcription_dir = os.path.join(output_dir, "transcriptions")
    os.makedirs(transcription_dir, exist_ok=True)

    # --- Discover audio files ---
    audio_files = {}
    for fname in sorted(os.listdir(enhanced_dir)):
        if fname.endswith(".wav"):
            label = fname.replace(".wav", "")
            # Clean up label
            parts = label.split("_", 1)
            if len(parts) == 2:
                label = parts[1].replace("_", " ").title()
                if "original" in fname.lower():
                    label = "Original (Noisy)"
            audio_files[label] = os.path.join(enhanced_dir, fname)

    if not audio_files:
        print("ERROR: No .wav files found in", enhanced_dir, file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files:")
    for label, path in audio_files.items():
        print(f"  • {label}: {path}")

    # --- Load signals ---
    signals = {}
    for label, path in audio_files.items():
        sig, _ = librosa.load(path, sr=sr, mono=True)
        signals[label] = sig.astype(np.float32)

    # --- SNR metrics ---
    print("\n" + "=" * 60)
    print("SNR ANALYSIS")
    print("=" * 60)

    baseline_key = "Original (Noisy)"
    if baseline_key not in signals:
        # Fallback: use first key
        baseline_key = list(signals.keys())[0]

    baseline_metrics = estimate_snr_metrics(signals[baseline_key], sr=sr)

    metrics_rows = []
    for name, sig in signals.items():
        m = estimate_snr_metrics(sig, sr=sr)
        row = {
            "Method": name,
            "Global SNR (dB)": round(m["global_snr_db"], 2),
            "Segmental SNR (dB)": round(m["segmental_snr_db"], 2),
            "Global Improvement (dB)": round(m["global_snr_db"] - baseline_metrics["global_snr_db"], 2),
            "Segmental Improvement (dB)": round(m["segmental_snr_db"] - baseline_metrics["segmental_snr_db"], 2),
            "Speech Frames (%)": round(m["speech_frames_pct"], 1),
            "RMS": round(float(np.sqrt(np.mean(sig ** 2))), 4),
            "Peak": round(float(np.max(np.abs(sig))), 4),
        }
        metrics_rows.append(row)
        print(f"\n  {name}:")
        for k, v in row.items():
            if k != "Method":
                print(f"    {k}: {v}")

    # Save CSV
    csv_path = os.path.join(output_dir, "enhancement_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\n  Saved SNR summary → {csv_path}")

    # --- Plots ---
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_waveforms(signals, sr, os.path.join(output_dir, "waveform_comparison.png"))
    plot_spectrograms(signals, sr, os.path.join(output_dir, "spectrogram_comparison.png"))

    # --- Whisper ASR ---
    print("\n" + "=" * 60)
    print("WHISPER ASR COMPARISON & WER")
    print("=" * 60)

    ground_truth = extract_ground_truth(vtt_path, start, duration)
    has_gt = bool(ground_truth.strip())
    
    if has_gt:
        print(f"Ground Truth loaded from VTT ({len(ground_truth.split())} words)")
        gt_path = os.path.join(transcription_dir, "ground_truth.txt")
        with open(gt_path, "w") as f:
            f.write(ground_truth)
    else:
        print(f"WARN: No ground truth matching time window ({start}s - {start+duration}s) found in {vtt_path}")

    asr_results = {}
    for label, path in audio_files.items():
        try:
            result = transcribe_with_whisper(path, model_name=whisper_model)
            
            # Compute WER if GT is available
            wer_score = None
            if has_gt and result["text"].strip():
                # Normalize text for fair WER comparison (lower, no punctuation)
                gt_norm = jiwer.RemovePunctuation()(jiwer.ToLowerCase()(ground_truth))
                hyp_norm = jiwer.RemovePunctuation()(jiwer.ToLowerCase()(result["text"]))
                wer_score = jiwer.wer(gt_norm, hyp_norm)
                result["wer"] = wer_score
                print(f"    → WER: {wer_score:.2%}")
                
            asr_results[label] = result

            # Save transcript
            safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            txt_path = os.path.join(transcription_dir, f"{safe_name}.txt")
            with open(txt_path, "w") as f:
                f.write(f"# Transcription: {label}\n")
                f.write(f"# Word count: {result['word_count']}\n")
                if wer_score is not None:
                    f.write(f"# WER vs Ground Truth: {wer_score:.2%}\n")
                f.write(f"# Processing time: {result['elapsed_s']}s\n\n")
                f.write(result["text"])
            print(f"    Saved transcript → {txt_path}")

        except Exception as exc:
            print(f"  ✗ ASR failed for {label}: {exc}")
            asr_results[label] = {"text": "", "word_count": 0, "wer": None, "error": str(exc)}

    # --- ASR comparison report ---
    report_path = os.path.join(output_dir, "asr_comparison.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("WHISPER ASR — BEFORE vs. AFTER ENHANCEMENT COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        baseline_asr = asr_results.get(baseline_key, {})
        baseline_words = baseline_asr.get("word_count", 0)

        f.write(f"Whisper model: {whisper_model}\n")
        if has_gt:
            f.write(f"Ground Truth baseline: {vtt_path} [{start}s - {start+duration}s]\n\n")
            f.write(f"{'Method':<35} {'Words':>8}  {'Δ Words':>8}  {'Time (s)':>10}  {'WER (%)':>10}\n")
            f.write("-" * 80 + "\n")
            for label, res in asr_results.items():
                wc = res.get("word_count", 0)
                delta = wc - baseline_words
                elapsed = res.get("elapsed_s", "N/A")
                wer = res.get("wer")
                wer_str = f"{wer:.2%}" if wer is not None else "N/A"
                f.write(f"{label:<35} {wc:>8}  {delta:>+8}  {elapsed:>10}  {wer_str:>10}\n")
        else:
            f.write(f"\n{'Method':<35} {'Words':>8}  {'Δ Words':>8}  {'Time (s)':>10}\n")
            f.write("-" * 70 + "\n")
            for label, res in asr_results.items():
                wc = res.get("word_count", 0)
                delta = wc - baseline_words
                elapsed = res.get("elapsed_s", "N/A")
                f.write(f"{label:<35} {wc:>8}  {delta:>+8}  {elapsed:>10}\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("TRANSCRIPTION EXCERPTS (first ~500 chars)\n")
        f.write("=" * 80 + "\n")
        
        if has_gt:
            gt_excerpt = ground_truth[:500] + ("…" if len(ground_truth) > 500 else "")
            f.write(f"\n--- GROUND TRUTH ---\n")
            f.write(textwrap.fill(gt_excerpt, width=80) + "\n")
            
        for label, res in asr_results.items():
            text = res.get("text", "(no transcription)")
            excerpt = text[:500] + ("…" if len(text) > 500 else "")
            f.write(f"\n--- {label} ---\n")
            f.write(textwrap.fill(excerpt, width=80) + "\n")

    print(f"\n  Saved ASR comparison report → {report_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}/")
    print(f"  • enhancement_summary.csv")
    print(f"  • waveform_comparison.png")
    print(f"  • spectrogram_comparison.png")
    print(f"  • asr_comparison.txt")
    print(f"  • transcriptions/ (one .txt per method)")
    print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="meow.m4a",
                   help="Path to the original source audio (default: meow.m4a)")
    p.add_argument("--enhanced-dir", default="enhanced_outputs",
                   help="Directory containing enhanced .wav files")
    p.add_argument("--vtt-file", default="meow.en.vtt",
                   help="Path to YouTube VTT subtitle file for ground truth")
    p.add_argument("--output-dir", default="analysis_results",
                   help="Output directory for results (default: analysis_results)")
    p.add_argument("--sr", type=int, default=16000,
                   help="Sample rate (default: 16000)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start offset in seconds")
    p.add_argument("--duration", type=float, default=180.0,
                   help="Duration in seconds")
    p.add_argument("--whisper-model", default="base",
                   help="Whisper model size (default: base)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        input_path=args.input,
        enhanced_dir=args.enhanced_dir,
        output_dir=args.output_dir,
        vtt_path=args.vtt_file,
        sr=args.sr,
        start=args.start,
        duration=args.duration,
        whisper_model=args.whisper_model,
    )
