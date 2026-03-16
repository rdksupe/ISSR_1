import numpy as np
import librosa
from sklearn.decomposition import NMF
from .base import BaseDenoiser

class NMFDenoiser(BaseDenoiser):
    @property
    def name(self) -> str:
        return "ML: NMF Denoising"

    @property
    def key(self) -> str:
        return "nmf"

    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        n_fft = 1024
        hop_length = 256
        n_components = 24

        S = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(S)
        phase = np.angle(S)

        model = NMF(
            n_components=n_components,
            init="nndsvda",
            max_iter=500,
            random_state=42,
            beta_loss="frobenius",
            solver="cd",
        )
        W = model.fit_transform(mag)
        H = model.components_

        comp_var = np.var(H, axis=1)
        keep = comp_var >= np.percentile(comp_var, 45)
        if keep.sum() == 0:
            keep[np.argmax(comp_var)] = True

        mag_speech = W[:, keep] @ H[keep, :]
        mag_speech = np.clip(mag_speech, 0.0, None)

        S_out = mag_speech * np.exp(1j * phase)
        y_out = librosa.istft(S_out, hop_length=hop_length, length=len(sig))
        return self.normalize(y_out)
