import numpy as np
import torch
from .base import BaseDenoiser

class MetaDenoiser(BaseDenoiser):
    @property
    def name(self) -> str:
        return "AI: Meta DNS64"

    @property
    def key(self) -> str:
        return "meta_denoise"

    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        from denoiser import pretrained
        model = pretrained.dns64().cpu()
        model.eval()

        wav = torch.from_numpy(sig).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            denoised = model(wav)[0, 0].numpy()
        return self.normalize(denoised)
