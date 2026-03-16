import numpy as np
import torch
import librosa
from .base import BaseDenoiser

class DeepFilterDenoiser(BaseDenoiser):
    @property
    def name(self) -> str:
        return "AI: DeepFilterNet3"

    @property
    def key(self) -> str:
        return "deepfilter"

    def enhance(self, sig: np.ndarray, sr: int) -> np.ndarray:
        from df.enhance import init_df, enhance
        
        target_sr = 48000
        if sr != target_sr:
            sig_48k = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
        else:
            sig_48k = sig

        model, df_state, _ = init_df()
        wav = torch.from_numpy(sig_48k).unsqueeze(0)
        enhanced_48k = enhance(model, df_state, wav)
        
        enhanced = enhanced_48k.cpu().numpy().squeeze()
        if sr != target_sr:
            enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=sr)
            
        return self.normalize(enhanced)
