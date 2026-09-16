"""
Audio feature extraction functions for numpy arrays.
"""

from typing import Literal

from einops import rearrange
import numpy as np
import torch
import torchaudio

from drum_gesture.core import OnsetDetection


class Loudness:
    def __init__(self, db: bool = False, eps: float = 1e-8):
        super().__init__()
        self.db = db
        self.eps = eps

    def __call__(self, frames: np.array):
        assert frames.ndim == 2

        # Calculate RMS
        rms = np.sqrt(np.mean(np.square(frames), axis=1))

        # Convert to dB
        if self.db:
            rms = 20 * np.log10(rms + self.eps)

        return rms


class SpectralCentroid:
    def __init__(self):
        pass

    def __call__(self, frames: np.array):
        assert frames.ndim == 2

        # Calculate FFT
        X = np.fft.rfft(frames, axis=1)
        X = np.abs(X)

        # Normalize -- using the torch version for compatibility
        X_norm = torch.nn.functional.normalize(torch.from_numpy(X), p=1, dim=-1).numpy()

        # Calculate spectral centroid
        bins = np.arange(X.shape[1])
        spectral_centroid = np.sum(bins * X_norm, axis=1)

        return spectral_centroid


class MelBand(torch.nn.Module):
    """
    Mel-frequency bands.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mels: int = 128,
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
        fft_size: int = None,
        window: Literal["hann", "flat_top", "none"] = "hann",
        mag_norm: bool = True,
        log_output: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.fft_size = fft_size
        self.window_fn = get_window_fn(window)
        self.mag_norm = mag_norm
        self.log_output = log_output

    @property
    def feature_names(self):
        return [f"mel_{i}" for i in range(self.n_mels)]

    def forward(self, x: torch.Tensor):
        window_size = x.shape[-1]

        # Apply a window
        if self.window_fn is not None:
            window = self.window_fn(x.shape[-1], device=x.device)
            x = x * window

        # Zero pad if necessary
        if self.fft_size is not None:
            pad_size = self.fft_size - x.shape[-1]
            pad_front = pad_size // 2
            pad_back = pad_size - pad_front
            x = torch.nn.functional.pad(x, (pad_front, pad_back))

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        # Rescale
        if self.mag_norm:
            X = X / (window_size / 4.0)

        fb = torchaudio.functional.melscale_fbanks(
            X.shape[-1], self.min_freq, self.max_freq, self.n_mels, self.sample_rate
        )
        mel_bands = torch.matmul(X, fb.to(X.device))

        # Normalize by magnitude
        if self.mag_norm:
            # Compute framewise energy
            energy = torch.sum(X, dim=-1, keepdim=True)
            energy = energy / (2.0 * self.fft_size / window_size)

            # Normalize by energy
            mel_sum = torch.sum(mel_bands, dim=-1, keepdim=True)
            mel_sum = torch.where(mel_sum < 1e-10, 1e-10, mel_sum)
            mel_bands = mel_bands * energy / mel_sum

        # Apply epsilon floor and convert to dB
        if self.log_output:
            mel_bands = torch.where(mel_bands < 1e-10, 1e-10, mel_bands)
            mel_bands = 20.0 * torch.log10(mel_bands)

        return mel_bands


class MelSpectrogram(torch.nn.Module):
    """
    Mel spectrogram feature extractor.
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 128,
        hop_size: int = 32,
        n_mels: int = 16,
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
        window: Literal["hann", "flat_top", "none"] = "hann",
        mag_norm: bool = True,
        log_output: bool = False,
    ):
        super().__init__()
        self.mel_band = MelBand(
            sample_rate=sample_rate,
            n_mels=n_mels,
            min_freq=min_freq,
            max_freq=max_freq,
            fft_size=n_fft,
            window=window,
            mag_norm=mag_norm,
            log_output=log_output,
        )
        self.n_fft = n_fft
        self.hop_size = hop_size

    # 20*log10(1e-10), the floor MelBand applies before taking the log.
    LOG_FLOOR_DB = -200.0

    def apply_gain(self, features: torch.Tensor, gain_db: float) -> torch.Tensor:
        """
        Apply a gain change directly in the feature domain.

        With ``mag_norm``, the mel bands are rescaled so their sum equals the frame
        energy. Scaling the input by ``g`` scales the bands, the energy and the band sum
        all by ``g``, so the output still scales linearly with ``g`` -- an additive dB
        offset once ``log_output`` has taken 20*log10.

        This is exact for every frame above the internal floors and wrong only for frames
        pinned at them, where the output is clamped rather than scaled. Verified against
        recomputing from gained audio in tests/test_gain_augmentation.py.
        """
        if self.mel_band.log_output:
            return torch.clamp(features + gain_db, min=self.LOG_FLOOR_DB)
        return features * (10.0 ** (gain_db / 20.0))

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, "Input must be a 2D tensor (channels, time)"
        assert x.shape[0] == 1, "Batch size must be 1 for MelSpectrogram"

        x = x.unsqueeze(0)  # Add batch dimension
        x = torch.nn.functional.unfold(
            x.unsqueeze(0),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_size),
            padding=(0, self.n_fft // 2),
        )

        x = rearrange(x, "b c t -> b t c").squeeze(0)  # Remove batch dimension

        mel_spectrogram = self.mel_band(x)
        return mel_spectrogram


class MFCC(torch.nn.Module):
    """
    Mel-frequency cepstral coefficients.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int = 13,
        n_mels: int = 128,
        fft_size: int = None,
        window: Literal["hann", "flat_top", "none"] = "hann",
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
        start_coeff: int = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.fft_size = fft_size
        self.window_fn = get_window_fn(window)
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.start_coeff = start_coeff

    @property
    def feature_names(self):
        return [f"mfcc_{i}" for i in range(self.start_coeff, self.n_mfcc)]

    def forward(self, x: torch.Tensor):
        # Apply a window
        if self.window_fn is not None:
            window = self.window_fn(x.shape[-1], device=x.device)
            x = x * window

        # Zero pad if necessary
        if self.fft_size is not None:
            pad_size = self.fft_size - x.shape[-1]
            pad_front = pad_size // 2
            pad_back = pad_size - pad_front
            x = torch.nn.functional.pad(x, (pad_front, pad_back))

        # Calculate FFT
        X = torch.fft.rfft(x, dim=-1)
        X = torch.abs(X)

        fb = torchaudio.functional.melscale_fbanks(
            X.shape[-1], self.min_freq, self.max_freq, self.n_mels, self.sample_rate
        )
        mel_scale = torch.matmul(X, fb.to(X.device))

        # Apply epsilon floor and convert to dB
        mel_scale = torch.where(mel_scale < 1e-10, 1e-10, mel_scale)
        mel_scale = 20.0 * torch.log10(mel_scale)

        dct_mat = torchaudio.functional.create_dct(
            self.n_mfcc, self.n_mels, norm="ortho"
        )
        mfcc = torch.matmul(mel_scale, dct_mat.to(mel_scale.device))

        return mfcc[..., self.start_coeff :]


class OnsetSignal(torch.nn.Module):
    """
    Onset signals extraction.
    """

    def __init__(
        self,
        sample_rate: int,
        hop_size: int,
        fast_env_attack: float,
        fast_env_release: float,
        slow_env_attack: float,
        slow_env_release: float,
        include_diff: bool,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.detection = OnsetDetection(
            sr=sample_rate,
            fast_env_attack=fast_env_attack,
            fast_env_release=fast_env_release,
            slow_env_attack=slow_env_attack,
            slow_env_release=slow_env_release,
        )
        self.include_diff = include_diff

    def apply_gain(self, features: torch.Tensor, gain_db: float) -> torch.Tensor:
        """
        Approximate a gain change directly in the feature domain.

        This is genuinely an approximation, not an identity. The pipeline floors the
        rectified signal at ``min_db`` *before* the envelope followers, so changing the
        gain changes which parts of the signal clip, and the asymmetric one-pole
        envelopes then follow a different trajectory rather than a shifted one. Measured
        against recomputing from gained audio on decaying noise bursts, frames above the
        floor still show around 1 dB of spread on the envelope channels, and the
        difference channel -- which would be exactly gain-invariant without the floor --
        moves by several dB.

        Whether that error matters is an empirical question about real recordings, whose
        noise floor sits well above digital silence. See
        scripts/compare_gain_augmentation.py.
        """
        offsets = torch.zeros(features.shape[-1], device=features.device)
        offsets[0] = gain_db  # env_fast
        offsets[1] = gain_db  # env_slow
        # offsets[2], the fast/slow difference, stays 0.

        shifted = features + offsets
        min_db = self.detection.min_db
        shifted[..., :2] = torch.clamp(shifted[..., :2], min=min_db)
        return shifted

    def forward(self, x: torch.Tensor):
        assert x.ndim == 2, "Input must be a 2D tensor (channels, time)"
        assert x.shape[0] == 1, "Batch size must be 1 for OnsetSignals"

        _, env_fast, env_slow = self.detection._onset_signal(x)

        # Downsample the env_fast and env_slow signals to hop_size
        env_fast = env_fast[:, :: self.hop_size]
        env_slow = env_slow[:, :: self.hop_size]

        env_fast = torch.from_numpy(env_fast).float()
        env_slow = torch.from_numpy(env_slow).float()
        envs = [env_fast, env_slow]

        if self.include_diff:
            env_diff = env_fast - env_slow
            envs.append(env_diff)

        # Stack the signals
        env = torch.vstack(envs).T

        return env


def get_window_fn(window: str):
    if window == "hann":
        return torch.hann_window
    elif window == "flat_top":
        return flat_top_window
    elif window == "none":
        return None
    else:
        raise ValueError(f"Unknown window type: {window}")


def flat_top_window(size, device="cpu"):
    """
    Flat top window for spectral analysis.
    https://en.wikipedia.org/wiki/Window_function#Flat_top_window
    """
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368

    n = torch.arange(size, dtype=torch.float, device=device)
    window = (
        a0
        - a1 * torch.cos(2 * torch.pi * n / (size - 1))
        + a2 * torch.cos(4 * torch.pi * n / (size - 1))
        - a3 * torch.cos(6 * torch.pi * n / (size - 1))
        + a4 * torch.cos(8 * torch.pi * n / (size - 1))
    )
    return window
