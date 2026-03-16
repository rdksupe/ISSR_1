import numpy as np
import noisereduce as nr
from .base import BaseDenoiser

class SpectralGateDenoiser(BaseDenoiser):
    @property
    def name(self) -> str:
        return "Classical: Spectral Gate"

    @property
    def key(self) -> str:
        return "spectral_gate"

    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        noise_clip = sig[: int(0.5 * sr)]
        out = nr.reduce_noise(
            y=sig,
            sr=sr,
            y_noise=noise_clip,
            stationary=False,
            prop_decrease=1.0,
        )
        return self.normalize(out)
