"""
Training directly from precomputed feature frames.

This is the path the Max4Live trainer will use. Rather than recording audio and
extracting features in Python, the device captures the feature frames its inference
front-end already produces (jem.onsetsignals, fluid.melbands~) alongside a label channel
driven live by a held footswitch or pad. Training then never touches audio.

Keeping the same path available in Python matters for two reasons: it gives power users
an escape hatch from the device, and it is what makes C++/Python parity testing possible,
since both sides can be fed byte-identical inputs.

A "take" is one continuous recording: an array of shape (time, channels) whose last
channel is the gesture activity target in [0, 1] and whose remaining channels are the
features. Takes are stored as multichannel float WAV (which ``buffer~ write`` produces
natively), .npy/.npz, or JSON.
"""

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset

from drum_gesture.data import IntervalCropSampler


def load_take(path: str, label_channel: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load one take, returning (features, labels).

    Args:
        path: a .wav (multichannel float), .npy, .npz or .json file.
        label_channel: which channel holds the target. Defaults to the last.

    Returns:
        features of shape (time, channels - 1) and labels of shape (time,).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".wav":
        import torchaudio

        # buffer~ write produces (channels, time); we want (time, channels).
        data, _ = torchaudio.load(str(path))
        data = data.T
    elif suffix == ".npy":
        data = torch.from_numpy(np.load(path)).float()
    elif suffix == ".npz":
        loaded = np.load(path)
        key = "frames" if "frames" in loaded else loaded.files[0]
        data = torch.from_numpy(loaded[key]).float()
    elif suffix == ".json":
        data = torch.tensor(_frames_from_json(path), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported take format: {path.suffix}")

    if data.ndim != 2:
        raise ValueError(
            f"{path}: expected 2D (time, channels), got shape {tuple(data.shape)}"
        )
    if data.shape[1] < 2:
        raise ValueError(
            f"{path}: needs at least one feature channel plus a label channel, "
            f"got {data.shape[1]}"
        )

    index = label_channel % data.shape[1]
    labels = data[:, index]
    features = torch.cat([data[:, :index], data[:, index + 1 :]], dim=1)
    return features, labels


def _frames_from_json(path: Path) -> List[List[float]]:
    """
    Read frames from JSON, accepting either a plain list of rows or a FluCoMa
    ``fluid.dataset~`` dump, whose rows are keyed by stringified integers.
    """
    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: cannot find frame rows in JSON")

    # A fluid.dataset~ is an unordered map, but this is a time series -- recover order
    # from the integer keys rather than trusting insertion order.
    try:
        keys = sorted(data.keys(), key=int)
    except ValueError as exc:
        raise ValueError(
            f"{path}: dataset keys are not integers, so frame order cannot be recovered"
        ) from exc

    return [data[k] for k in keys]


class FrameGestureDataset(Dataset):
    """
    Random crops of precomputed feature frames.

    Mirrors ContinuousGestureDataset, but its timeline is measured in feature frames
    rather than audio samples and it has no feature extractor.
    """

    def __init__(
        self,
        features: torch.Tensor,  # (time, channels)
        labels: torch.Tensor,  # (time,)
        segment_frames: int,
        num_data: int = 1000,
        standardize: bool = True,
        random_gain: float = 0.0,  # dB, applied via gain_offsets
        gain_offsets: Optional[torch.Tensor] = None,  # per-channel dB offset per 1 dB
        intervals: Optional[List[Tuple[int, int]]] = None,
        deterministic: bool = False,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features and labels disagree on length: "
                f"{features.shape[0]} vs {labels.shape[0]}"
            )

        self.features = features
        self.labels = labels
        self.segment_frames = segment_frames
        self.num_data = num_data
        self.deterministic = deterministic
        self.random_gain = random_gain
        self.gain_offsets = gain_offsets

        if intervals is None:
            intervals = [(0, features.shape[0])]
        self.sampler = IntervalCropSampler(
            intervals,
            span=segment_frames,
            num_data=num_data,
            deterministic=deterministic,
            what="frame",
        )

        self.standardize = standardize
        if self.standardize:
            if mean is not None and std is not None:
                self.mean, self.std = mean, std
            else:
                self.fit_standardize()

    @property
    def intervals(self) -> List[Tuple[int, int]]:
        return self.sampler.intervals

    def fit_standardize(self):
        frames = torch.cat([self.features[s:e] for s, e in self.intervals], dim=0)
        self.mean = frames.mean(dim=0)
        self.std = frames.std(dim=0)
        # A channel that never varies would otherwise produce NaNs downstream.
        self.std = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)
        logger.info(f"Fitted standardization: mean={self.mean}, std={self.std}")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        start = self.sampler.start(idx)
        end = start + self.segment_frames

        features = self.features[start:end]
        labels = self.labels[start:end]

        if (
            self.random_gain > 0
            and not self.deterministic
            and self.gain_offsets is not None
        ):
            gain_db = (torch.randn(1) * self.random_gain / 2.0).item()
            features = features + self.gain_offsets * gain_db

        if self.standardize:
            features = (features - self.mean) / self.std

        return features, labels


def create_frame_gesture_datasets(
    take_files: Sequence[str],
    segment_frames: int,
    num_data: int = 1000,
    val_fraction: float = 0.2,
    val_num_data: Optional[int] = None,
    label_channel: int = -1,
    random_gain: float = 0.0,
    gain_offsets: Optional[Sequence[float]] = None,
) -> Tuple[FrameGestureDataset, Optional[FrameGestureDataset]]:
    """
    Load takes and build train/validation datasets with a leak-free split.

    The last ``val_fraction`` of each take is held out, matching the audio path.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not take_files:
        raise ValueError("No take files given")

    all_features = []
    all_labels = []
    boundaries = []
    offset = 0

    for path in take_files:
        features, labels = load_take(path, label_channel=label_channel)
        logger.info(
            f"Loaded {path}: {features.shape[0]} frames, {features.shape[1]} channels, "
            f"{labels.mean().item():.1%} gesture-active"
        )
        all_features.append(features)
        all_labels.append(labels)
        boundaries.append((offset, offset + features.shape[0]))
        offset += features.shape[0]

    channels = {f.shape[1] for f in all_features}
    if len(channels) != 1:
        raise ValueError(f"Takes disagree on channel count: {sorted(channels)}")

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    offsets_tensor = None
    if gain_offsets is not None:
        offsets_tensor = torch.tensor(gain_offsets, dtype=torch.float32)
        if offsets_tensor.shape[0] != features.shape[1]:
            raise ValueError(
                f"gain_offsets has {offsets_tensor.shape[0]} entries but there are "
                f"{features.shape[1]} feature channels"
            )

    train_intervals = []
    val_intervals = []
    for start, end in boundaries:
        split = end - int((end - start) * val_fraction)
        train_intervals.append((start, split))
        if val_fraction > 0:
            val_intervals.append((split, end))

    train = FrameGestureDataset(
        features=features,
        labels=labels,
        segment_frames=segment_frames,
        num_data=num_data,
        random_gain=random_gain,
        gain_offsets=offsets_tensor,
        intervals=train_intervals,
    )

    if val_fraction == 0:
        return train, None

    val = FrameGestureDataset(
        features=features,
        labels=labels,
        segment_frames=segment_frames,
        num_data=val_num_data if val_num_data is not None else max(num_data // 5, 1),
        random_gain=random_gain,
        gain_offsets=offsets_tensor,
        intervals=val_intervals,
        deterministic=True,
        mean=train.mean,
        std=train.std,
    )

    def positive_fraction(intervals):
        total = sum(e - s for s, e in intervals)
        active = sum(labels[s:e].sum().item() for s, e in intervals)
        return active / total if total else 0.0

    val_pos = positive_fraction(val_intervals)
    logger.info(
        f"Split: train {len(train)} crops ({positive_fraction(train_intervals):.1%} "
        f"gesture-active), val {len(val)} crops ({val_pos:.1%} gesture-active)"
    )
    if val_pos == 0.0:
        logger.warning(
            "Validation split contains no gesture activity; val_loss will not be a "
            "useful early stopping signal."
        )

    return train, val
