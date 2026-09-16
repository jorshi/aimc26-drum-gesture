from pathlib import Path
from typing import Optional, Union

import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from scipy.io import wavfile
import torch
import torchaudio


def to_absolute_dataset_path(path: Union[str, None], cfg: DictConfig) -> str:
    """
    Convert a relative path to an absolute path based on the dataset directory in the configuration.
    """
    if path is None:
        return None
    return hydra.utils.to_absolute_path(str(Path(cfg.data_dir) / path))


def load_audio(
    path: Path,
    sample_rate: int,
    max_seconds: Optional[float] = None,
    normalize: bool = False,
):
    input_sr, data = wavfile.read(path)

    # Convert integer audio to float32
    data_info = np.iinfo(data.dtype)
    data = data.astype(np.float32, order="C")
    data = data / abs(data_info.min)
    data = torch.from_numpy(data)
    if data.ndim == 2:
        data = data.T
        data = data[:1, ...]
    elif data.ndim == 1:
        data = data[None, ...]
    else:
        raise RuntimeError(f"Unexpected audio input shape {data.shape}")

    if input_sr != sample_rate:
        logger.info(f"Resampling audio from {input_sr} to {sample_rate}")
        data = torchaudio.functional.resample(
            data, input_sr, sample_rate, lowpass_filter_width=512
        )

    if max_seconds is not None:
        num_samples = int(max_seconds * sample_rate)
        data = data[:, :num_samples]

    if normalize:
        data = data / torch.max(torch.abs(data))

    return data, input_sr
