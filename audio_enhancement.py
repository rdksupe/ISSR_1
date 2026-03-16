#!/usr/bin/env python3
"""
Audio Enhancement Script — Apollo 13 Flight Director Loop (Modular)
====================================================================
Applies multiple denoising methods to a segment of the mission audio
and saves enhanced outputs for downstream analysis and ASR comparison.

Usage:
  python audio_enhancement.py [--input FILE] [--start SEC] [--duration SEC] [--methods KEY1,KEY2]
"""

import argparse
import os
import sys
import time
import numpy as np
import librosa
import soundfile as sf

from denoisers import discover_denoisers

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_audio(sig: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Peak-normalize to *peak* amplitude."""
    sig = np.asarray(sig, dtype=np.float32)
    m = np.max(np.abs(sig)) + 1e-9
    return (sig / m) * peak


def load_segment(path: str, sr: int, start: float, duration: float) -> tuple:
    """Load a mono segment from *path* at sample-rate *sr*."""
    print(f"Loading {duration:.0f}s segment starting at {start:.0f}s from: {path}")
    sig, sr_out = librosa.load(path, sr=sr, mono=True,
                               offset=start, duration=duration)
    sig = sig.astype(np.float32)
    actual_dur = len(sig) / sr_out
    print(f"  → Loaded {actual_dur:.2f}s @ {sr_out} Hz  ({len(sig)} samples)")
    return sig, sr_out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_enhancement(input_path: str, start: float, duration: float,
                    sr: int, output_dir: str, selected_methods: list = None) -> dict:
    """Run specified enhancement methods and save outputs."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Discover all available plugins
    available_denoisers = discover_denoisers()
    
    # 2. Filter methods if requested
    if selected_methods:
        # Validate keys
        invalid = [m for m in selected_methods if m not in available_denoisers]
        if invalid:
            print(f"ERROR: Invalid method keys: {invalid}")
            print(f"Available keys: {list(available_denoisers.keys())}")
            sys.exit(1)
        methods_to_run = {k: available_denoisers[k] for k in selected_methods}
    else:
        methods_to_run = available_denoisers

    print(f"\nDiscovered {len(available_denoisers)} models. Planning to run: {list(methods_to_run.keys())}")

    # 3. Load audio segment
    noisy_signal, sr = load_segment(input_path, sr, start, duration)

    # 4. Save normalized original for fair comparison (if it doesn't exist or always for consistency)
    orig_path = os.path.join(output_dir, "00_original.wav")
    orig_norm = normalize_audio(noisy_signal)
    sf.write(orig_path, orig_norm, sr)
    print(f"\nSaved normalized original → {orig_path}")

    results = {"Original (Noisy)": {"signal": orig_norm, "path": orig_path}}

    # 5. Execute each denoiser
    for idx, (key, denoiser) in enumerate(methods_to_run.items(), start=1):
        print(f"\n[{idx}/{len(methods_to_run)}] Running {denoiser.name} ({key}) …")
        t0 = time.time()
        try:
            enhanced = denoiser.enhance(noisy_signal, sr)
            elapsed = time.time() - t0
            
            out_path = os.path.join(output_dir, f"{idx:02d}_{key}.wav")
            sf.write(out_path, enhanced, sr)
            
            results[denoiser.name] = {"signal": enhanced, "path": out_path}
            print(f"  ✓ Done in {elapsed:.1f}s → {out_path}")

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ✗ FAILED after {elapsed:.1f}s — {exc}")
            results[denoiser.name] = {"signal": None, "path": None, "error": str(exc)}

    print(f"\n{'='*60}")
    print(f"Enhancement complete. Results in: {output_dir}")
    print(f"{'='*60}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="meow.m4a",
                   help="Path to the source audio file (default: meow.m4a)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start offset in seconds (default: 0)")
    p.add_argument("--duration", type=float, default=180.0,
                   help="Duration in seconds (default: 180 = 3 min)")
    p.add_argument("--sr", type=int, default=16000,
                   help="Target sample rate (default: 16000)")
    p.add_argument("--output-dir", default="enhanced_outputs",
                   help="Output directory (default: enhanced_outputs)")
    p.add_argument("--methods", type=str,
                   help="Comma-separated list of method keys to run (e.g., 'wiener,deepfilter'). Runs all if omitted.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Audio file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    selected = None
    if args.methods:
        selected = [m.strip() for m in args.methods.split(",") if m.strip()]

    run_enhancement(
        input_path=args.input,
        start=args.start,
        duration=args.duration,
        sr=args.sr,
        output_dir=args.output_dir,
        selected_methods=selected
    )
