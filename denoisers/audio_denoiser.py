import numpy as np
import torch
from .base import BaseDenoiser

class AudioDenoiserPlugin(BaseDenoiser):
    @property
    def name(self) -> str:
        return "AI: audio-denoiser (2024)"

    @property
    def key(self) -> str:
        return "audio_denoiser"

    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        from audio_denoiser.AudioDenoiser import AudioDenoiser
        denoiser = AudioDenoiser(device="cuda" if torch.cuda.is_available() else "cpu")
        
        wav = torch.from_numpy(sig).unsqueeze(0)
        out_sig = denoiser.process_waveform(wav, sample_rate=sr, return_cpu_tensor=True)
        return self.normalize(out_sig.numpy().squeeze())
