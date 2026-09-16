"""
Evaluation dataset for onset detection.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from numba import jit
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

from drum_gesture.utils import load_audio


class OnsetEvalDataset(Dataset):
    def __init__(
        self,
        audio_files: List[str],
        annotation_files: List[str],
        sample_rate: Optional[int] = None,
        channel: int = -1,
    ):
        """
        Initialize the dataset with a list of audio files.

        Args:
            audio_files (list): List of paths to audio files.
            annotation_files (list): List of paths to annotation files paired with audio files.
            sample_rate (int): Sample rate for loading audio files.
        """
        self.audio_files = audio_files
        self.annotation_files = annotation_files
        assert len(audio_files) == len(
            annotation_files
        ), "Number of audio files must match number of annotation files"
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.audio_files[idx])
        if self.sample_rate is not None and sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.sample_rate,
                lowpass_filter_width=128,
            )
        assert waveform.dim() == 2, "Audio waveform must be 2D (channels, samples)"
        timestamps = self.get_annotations(idx)
        return waveform[-1][None, ...], timestamps, sr, Path(self.audio_files[idx]).stem

    def get_annotations(self, idx):
        """
        Get annotations for the audio file at the specified index.

        Args:
            idx (int): Index of the audio file.

        Returns:
            List: List of annotations for the audio file.
        """
        with open(self.annotation_files[idx], "r") as f:
            annotations = f.readlines()

        timestamps = []
        for i, line in enumerate(annotations):
            timestamp = float(line.split(",")[0])
            timestamps.append(timestamp)

        return timestamps


def create_onset_eval_dataset(
    audio_dir: str,
    annotation_dir: List[str],
    sample_rate: Optional[int] = None,
    channel: Optional[int] = -1,
) -> OnsetEvalDataset:
    """
    Create an onset evaluation dataset.
    """

    # Find all audio files in the directory
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    annotation_files = []
    for audio_file in audio_files:
        annotation_file = Path(annotation_dir) / (audio_file.stem + ".csv")
        if annotation_file.exists():
            annotation_files.append(str(annotation_file))
        else:
            raise FileNotFoundError(f"Annotation file {annotation_file} not found.")

    audio_files = [str(file) for file in audio_files]
    return OnsetEvalDataset(
        audio_files=audio_files,
        annotation_files=annotation_files,
        sample_rate=sample_rate,
        channel=channel,
    )


class IntervalCropSampler:
    """
    Picks crop start positions from a set of allowed (start, end) ranges.

    Restricting crops to explicit intervals is what makes an honest train/validation split
    possible: both the audio corpus and a set of recorded takes are stored as one
    concatenated timeline, so sampling freely across all of it would put near-identical
    overlapping windows on both sides of the split.

    Starts are drawn uniformly over all valid positions, so longer intervals contribute
    proportionally more crops.
    """

    def __init__(
        self,
        intervals: List[Tuple[int, int]],
        span: int,
        num_data: int,
        deterministic: bool = False,
        what: str = "segment",
    ):
        self.span = span
        self.intervals = [(s, e) for s, e in intervals if e - s >= span]
        if not self.intervals:
            longest = max((e - s for s, e in intervals), default=0)
            raise ValueError(
                f"No interval is long enough for a {span}-{what} crop; the longest is "
                f"{longest} {what}s. Record longer takes, shorten segment_duration, or "
                f"lower val_fraction."
            )

        starts_per_interval = [(e - s) - span + 1 for s, e in self.intervals]
        self._cumulative = torch.tensor(starts_per_interval).cumsum(0)
        self._total_starts = int(self._cumulative[-1].item())

        self.deterministic = deterministic
        self._fixed_starts = None
        if deterministic:
            # Evenly spaced crops so validation loss is comparable across epochs.
            self._fixed_starts = [
                self._from_offset(int(i * self._total_starts / max(num_data, 1)))
                for i in range(num_data)
            ]

    def _from_offset(self, offset: int) -> int:
        """
        Map a flat index over all valid crop starts onto an absolute position.
        """
        offset = min(offset, self._total_starts - 1)
        interval_idx = int(torch.searchsorted(self._cumulative, offset, right=True))
        prior = (
            0 if interval_idx == 0 else int(self._cumulative[interval_idx - 1].item())
        )
        return self.intervals[interval_idx][0] + (offset - prior)

    def start(self, idx: int) -> int:
        if self.deterministic:
            return self._fixed_starts[idx]
        return self._from_offset(torch.randint(0, self._total_starts, (1,)).item())


class ContinuousGestureDataset(Dataset):
    """
    Random crops of a concatenated audio corpus, with frame-rate gesture activity targets.

    Crops are drawn from ``intervals``, a list of (start, end) sample ranges -- see
    ``IntervalCropSampler``.
    """

    def __init__(
        self,
        audio: torch.Tensor,  # audio data tensor
        true_labels: torch.Tensor,  # true labels tensor
        feature: torch.nn.Module,  # feature extractor
        sample_rate: int,  # sample rate of the audio
        segment_duration: float = 2.0,  # duration of each segment in seconds
        num_data: int = 1000,
        standardize: bool = True,
        random_silence_prob: float = 0.0,  # probability of inserting random silence
        random_gain: float = 12.0,  # random gain in dB
        gain_domain: str = "audio",  # "audio" or "feature"
        intervals: Optional[List[Tuple[int, int]]] = None,
        deterministic: bool = False,  # fixed crops, for validation
        mean: Optional[torch.Tensor] = None,  # reuse stats fitted elsewhere
        std: Optional[torch.Tensor] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.audio = audio
        self.true_labels = true_labels
        self.feature = feature
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.random_silence_prob = random_silence_prob
        self.random_gain = random_gain
        if gain_domain not in ("audio", "feature"):
            raise ValueError(
                f"gain_domain must be 'audio' or 'feature', got {gain_domain!r}"
            )
        if (
            gain_domain == "feature"
            and random_gain > 0
            and not hasattr(feature, "apply_gain")
        ):
            raise ValueError(
                f"{type(feature).__name__} does not implement apply_gain, so "
                "gain_domain='feature' is not available for it"
            )
        self.gain_domain = gain_domain
        self.num_data = num_data
        assert (
            self.audio.shape == self.true_labels.shape
        ), "Audio and labels must have the same shape"
        self.return_audio = False
        self.deterministic = deterministic
        self.seed = seed

        self.segment_samples = int(self.segment_duration * self.sample_rate)
        if intervals is None:
            intervals = [(0, self.audio.shape[-1])]
        self.sampler = IntervalCropSampler(
            intervals,
            span=self.segment_samples,
            num_data=self.num_data,
            deterministic=self.deterministic,
            what="sample",
        )

        # Learn standardization values
        self.standardize = standardize
        if self.standardize:
            if mean is not None and std is not None:
                self.mean, self.std = mean, std
            else:
                self.fit_standarize()

        # TODO make sure all audio input is normalized to -3dB? Alhough I have done this
        # during data collection. So I'm not sure whether this is strictly necessary.

    @property
    def intervals(self) -> List[Tuple[int, int]]:
        return self.sampler.intervals

    def fit_standarize(self):
        """
        Fit per-dimension mean/std over the dataset's own intervals only.

        Fitting over the whole corpus would leak validation statistics into training.
        """
        features = torch.cat(
            [self.feature(self.audio[:, s:e]) for s, e in self.intervals], dim=0
        )
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0)
        logger.info(
            "Fitted standardization values: mean={}, std={}".format(self.mean, self.std)
        )

    def __len__(self):
        # Placeholder for dataset length
        return self.num_data

    def __getitem__(self, idx):
        segment_start = self.sampler.start(idx)
        segment_end = segment_start + self.segment_samples
        audio_segment = self.audio[:, segment_start:segment_end]
        label_segment = self.true_labels[:, segment_start:segment_end]

        if self.random_silence_prob > 0:
            raise NotImplementedError(
                "Random silence insertion is not implemented yet."
            )

        # Augmentation is training-only; validation crops must be reproducible.
        augment = self.random_gain > 0 and not self.deterministic
        gain_db = None
        if augment:
            gain_db = (torch.randn(1) * self.random_gain / 2.0).item()
            if self.gain_domain == "audio":
                audio_segment = audio_segment * (10.0 ** (gain_db / 20.0))

        # Apply feature extraction
        features_segment = self.feature(audio_segment)

        # The C++ trainer works from precomputed feature frames and cannot recompute
        # features from audio, so it can only augment here. Training the released Python
        # models the same way keeps the two paths comparable.
        if augment and self.gain_domain == "feature":
            features_segment = self.feature.apply_gain(features_segment, gain_db)

        if self.standardize:
            features_segment = (features_segment - self.mean) / self.std

        # Interpolate labels to match the feature extraction output
        segment_true_signal = (
            torch.nn.functional.interpolate(
                label_segment.unsqueeze(0),
                size=(features_segment.shape[0],),
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )

        assert (
            features_segment.shape[0] == segment_true_signal.shape[0]
        ), "Feature segment and label segment must have the same length after interpolation."

        if self.return_audio:
            return features_segment, segment_true_signal, audio_segment

        return features_segment, segment_true_signal


def get_annotations(
    file_path: str, input_sr: Optional[int] = None, output_sr: Optional[int] = None
) -> list:
    """
    Get annotations for the audio file at the specified index.

    Args:
        idx (int): Index of the audio file.

    Returns:
        List: List of annotations for the audio file.
    """
    if file_path is None:
        return [], []
    else:
        file_path = Path(file_path)
        assert file_path.exists(), f"Annotation file does not exist: {file_path}"

    with open(file_path, "r") as f:
        annotations = f.readlines()

    if input_sr is not None:
        assert (
            output_sr is not None
        ), "If input_sr is provided, output_sr must also be provided."

    timestamps = []
    durations = []
    for i, line in enumerate(annotations):
        timestamp = float(line.split(",")[0])
        if input_sr is not None and output_sr is not None:
            timestamp = timestamp * (output_sr / input_sr)
        timestamps.append(timestamp)

        duration = float(line.split(",")[2])
        if input_sr is not None and output_sr is not None:
            duration = duration * (output_sr / input_sr)
        durations.append(duration)

    return timestamps, durations


@jit(nopython=True)
def make_annotation_signal(
    audio_length: int, annotations: Tuple[List, List]
) -> np.ndarray:
    """
    Create a signal from the annotations.
    """
    y_true = np.zeros(audio_length, dtype=np.float32)
    if len(annotations[0]) == 0:
        return y_true[None, :]

    n = 0
    for i in range(audio_length):
        current_start = annotations[0][n]
        current_end = current_start + annotations[1][n]
        if i >= current_start and i < current_end:
            y_true[i] = 1.0
        elif i >= current_end:
            n += 1
            if n >= len(annotations[0]):
                break

    return y_true[None, :]


def get_annotation_signal(
    audio: torch.Tensor, annotations: Tuple[List, List]
) -> torch.Tensor:
    """
    Create a signal from the annotations.

    Args:
        audio (torch.Tensor): Audio tensor.
        annotations (list): List of annotations.

    Returns:
        torch.Tensor: Signal tensor with 1s at annotation positions and 0s elsewhere.
    """
    y_true = make_annotation_signal(audio.shape[-1], annotations)
    y_true = torch.tensor(y_true, dtype=torch.float32)
    return y_true


def create_continuous_gesture_dataset(
    audio_files: List[str],
    annotation_files: List[str],
    feature: torch.nn.Module,
    sample_rate: int,
    num_data: int = 1000,
) -> ContinuousGestureDataset:
    """
    Create a continuous gesture dataset over the whole corpus, with no validation split.

    Kept for analysis code that wants a single dataset covering everything. Training
    should use ``create_continuous_gesture_datasets``.
    """
    audio, annotations, _ = _load_continuous_corpus(
        audio_files, annotation_files, sample_rate
    )

    # Create the dataset
    dataset = ContinuousGestureDataset(
        audio=audio,
        true_labels=annotations,
        feature=feature,
        sample_rate=sample_rate,
        segment_duration=1.0,  # Default segment duration in seconds
        random_silence_prob=0.0,  # Default random silence probability
        num_data=num_data,
    )

    return dataset


def _load_continuous_corpus(
    audio_files: List[str],
    annotation_files: List[str],
    sample_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Load and concatenate the corpus, returning per-source-file sample boundaries.
    """
    audio = []
    annotations = []
    boundaries = []
    offset = 0

    for audio_file, annotation_file in zip(audio_files, annotation_files):
        logger.info(f"Loading audio file: {audio_file}")
        waveform, sr = load_audio(audio_file, sample_rate)
        # waveform, sr = torchaudio.load(audio_file) 
    
        assert waveform.dim() == 2, "Audio waveform must be 2D (channels, samples)"
        if waveform.shape[0] > 1:
            waveform = waveform[:1, ...]

        audio.append(waveform)
        boundaries.append((offset, offset + waveform.shape[-1]))
        offset += waveform.shape[-1]

        logger.info(f"Loading annotations from: {annotation_file}")
        annos = get_annotations(annotation_file, input_sr=sr, output_sr=sample_rate)
        assert len(annos[0]) == len(annos[1]), "Annotations must have the same length"
        if len(annos[0]) == 0:
            logger.warning(
                f"No annotations found for {audio_file}. Using empty annotations."
            )
            annos = torch.zeros((1, waveform.shape[-1]), dtype=torch.float32)
        else:
            annos = get_annotation_signal(waveform, annos)
        annotations.append(annos)

    return torch.cat(audio, dim=-1), torch.cat(annotations, dim=-1), boundaries


def create_continuous_gesture_datasets(
    audio_files: List[str],
    annotation_files: List[str],
    feature: torch.nn.Module,
    sample_rate: int,
    num_data: int = 1000,
    val_fraction: float = 0.2,
    val_num_data: Optional[int] = None,
    segment_duration: float = 1.0,
    random_gain: float = 12.0,
    gain_domain: str = "audio",
    seed: int = 0,
) -> Tuple[ContinuousGestureDataset, Optional[ContinuousGestureDataset]]:
    """
    Build train and validation datasets with a leak-free split.

    The last ``val_fraction`` of each source file is held out. Splitting within each file
    rather than holding out whole files keeps the positive/negative balance of the split
    close to the corpus as a whole -- several files in the study corpus are entirely
    negative, so a file-level split would easily produce a validation set with no gesture
    in it at all.

    Standardization statistics are fitted on the training intervals only and shared with
    the validation set.

    Returns ``(train, None)`` when ``val_fraction`` is 0.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    audio, annotations, boundaries = _load_continuous_corpus(
        audio_files, annotation_files, sample_rate
    )
    logger.info(
        f"Corpus: {audio.shape[-1] / sample_rate:.1f}s across {len(boundaries)} files"
    )

    train_intervals = []
    val_intervals = []
    for start, end in boundaries:
        split = end - int((end - start) * val_fraction)
        train_intervals.append((start, split))
        if val_fraction > 0:
            val_intervals.append((split, end))

    train = ContinuousGestureDataset(
        audio=audio,
        true_labels=annotations,
        feature=feature,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        num_data=num_data,
        random_gain=random_gain,
        gain_domain=gain_domain,
        intervals=train_intervals,
        seed=seed,
    )

    if val_fraction == 0:
        return train, None

    val = ContinuousGestureDataset(
        audio=audio,
        true_labels=annotations,
        feature=feature,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        num_data=val_num_data if val_num_data is not None else max(num_data // 5, 1),
        random_gain=random_gain,
        gain_domain=gain_domain,
        intervals=val_intervals,
        deterministic=True,
        mean=train.mean,
        std=train.std,
        seed=seed,
    )

    _log_split_balance(train, val, annotations, train_intervals, val_intervals)
    return train, val


def _log_split_balance(train, val, annotations, train_intervals, val_intervals):
    """
    Report how much gesture activity landed on each side of the split.

    A validation set with no positive frames makes val_loss meaningless as an early
    stopping signal, and with a musician-scale corpus that is a real possibility.
    """

    def positive_fraction(intervals):
        total = sum(e - s for s, e in intervals)
        if total == 0:
            return 0.0
        active = sum(annotations[0, s:e].sum().item() for s, e in intervals)
        return active / total

    train_pos = positive_fraction(train_intervals)
    val_pos = positive_fraction(val_intervals)
    logger.info(
        f"Split: train {len(train)} crops ({train_pos:.1%} gesture-active), "
        f"val {len(val)} crops ({val_pos:.1%} gesture-active)"
    )
    if val_pos == 0.0:
        logger.warning(
            "Validation split contains no gesture activity; val_loss will not be a "
            "useful early stopping signal."
        )
