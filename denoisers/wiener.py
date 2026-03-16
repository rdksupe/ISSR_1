import numpy as np
from scipy.signal import wiener
from .base import BaseDenoiser

class WienerDenoiser(BaseDenoiser):
    @property
    def name(self) -> str:
        return "Classical: Wiener Filter"

    @property
    def key(self) -> str:
        return "wiener"

    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        out = wiener(sig, mysize=29)
        return self.normalize(out)
