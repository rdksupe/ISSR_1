import numpy as np
from abc import ABC, abstractmethod

class BaseDenoiser(ABC):
    """Base class for all audio denoiser plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the denoiser."""
        pass

    @property
    @abstractmethod
    def key(self) -> str:
        """Unique identifier key for the denoiser (used in filename and --methods)."""
        pass

    @abstractmethod
    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply enhancement to the input signal.
        @param sig: 1D numpy array (mono audio)
        @param sr: Sampling rate
        @return: Enhanced 1D numpy array
        """
        pass

    def normalize(self, sig: np.ndarray, peak: float = 0.98) -> np.ndarray:
        """Common peak-normalization helper."""
        sig = np.asarray(sig, dtype=np.float32)
        m = np.max(np.abs(sig)) + 1e-9
        return (sig / m) * peak
