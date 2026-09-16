"""
Evaluation module for onset detection.
Class that receives a dataset and onset detection method, and evaluates the
method on the dataset.
"""

from typing import List, Callable, Literal
from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass
class OnsetEvaluationResult:
    """
    Data class to store the results from onset detection evaluation.
    """

    true_positives: float
    false_positives: float
    false_negatives: float

    @property
    def precision(self) -> float:
        """Calculate precision."""
        return (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )

    @property
    def recall(self) -> float:
        """Calculate recall."""
        return (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        precision = self.precision
        recall = self.recall
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    def __str__(self) -> str:
        """String representation of the evaluation results."""
        return (
            f"True Positives: {self.true_positives}, "
            f"False Positives: {self.false_positives}, "
            f"False Negatives: {self.false_negatives}, "
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}, "
            f"F1 Score: {self.f1_score:.4f}"
        )

    def get_as_dict(self) -> dict:
        """Get the evaluation results as a dictionary."""
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


class OnsetEvaluation:
    def __init__(
        self,
        dataset: Dataset,
        detector: Callable,
        method: Literal["accurary"] = "accuracy",
        tolerance: float = 0.003,
    ):
        """
        Initialize the evaluation module.

        Args:
            dataset (Dataset): Dataset containing audio files and annotations.
            detector (Callable): Onset detection method to evaluate.

        TODO: Add macro-averaging and micro-averaging options.
        """
        self.dataset = dataset
        self.detector = detector
        self.method = method
        self.tolerance = tolerance

    def evaluate(self) -> OnsetEvaluationResult:
        """
        Evaluate the onset detection method on the dataset.

        Returns:
            OnsetEvaluationResult: Evaluation results containing true positives,
                false positives, and false negatives.
        """
        results = {}
        for audio, annotations, sr, file_name in self.dataset:
            assert (
                audio.ndim == 2 and audio.shape[0] == 1
            ), "Audio must be monophonic (1 channel)."
            # Update the detector to handle the audio input
            self.detector.update_sample_rate(sr)

            # Detect onsets using the provided detector
            detected_onsets = self.detector(audio)

            # Convert onsets from samples to seconds
            detected_onsets = onset_samples_to_seconds(detected_onsets, sr)
            annotations = onset_samples_to_seconds(annotations, sr)

            # Calculate metrics
            result = self._calculate_metrics(detected_onsets, annotations)
            results[file_name] = result

        # Aggregate results across all files
        aggregated_result = self._aggregate_results(results)
        return aggregated_result, results

    def _calculate_metrics(
        self, detected_onsets: List[float], annotations: List[float]
    ) -> OnsetEvaluationResult:
        """
        Calculate metrics for the detected onsets against the annotations.

        Args:
            detected_onsets (List[float]): List of detected onset times in seconds.
            annotations (List[float]): List of true onset times in seconds.

        Returns:
            OnsetEvaluationResult: Evaluation results containing true positives,
                false positives, and false negatives.
        """
        if self.method == "accuracy":
            # Use the accuracy metric for evaluation
            return accuracy_metric(detected_onsets, annotations, self.tolerance)
        else:
            raise ValueError(f"Unknown evaluation method: {self.method}")

    def _aggregate_results(self, results: dict) -> OnsetEvaluationResult:
        """
        Aggregate results from multiple files into a single evaluation result.

        Args:
            results (dict): Dictionary of evaluation results for each file.

        Returns:
            OnsetEvaluationResult: Aggregated evaluation results.
        """
        total_true_positives = sum(result.true_positives for result in results.values())
        total_false_positives = sum(
            result.false_positives for result in results.values()
        )
        total_false_negatives = sum(
            result.false_negatives for result in results.values()
        )

        return OnsetEvaluationResult(
            true_positives=total_true_positives,
            false_positives=total_false_positives,
            false_negatives=total_false_negatives,
        )


def accuracy_metric(
    detected_onsets: List[float], true_onsets: List[float], tolerance: float = 0.003
) -> OnsetEvaluationResult:
    """
    Calculate accuracy metrics for onset detection.

    Args:
        detected_onsets (List[float]): List of detected onset samples.
        true_onsets (List[float]): List of true onset samples.
        tolerance (float): Tolerance for matching onsets.

    Returns:
        OnsetEvaluationResult: Evaluation results containing true positives,
            false positives, and false negatives.
    """
    matches = {}
    for i in range(len(true_onsets)):
        for j in range(len(detected_onsets)):
            difference = abs(detected_onsets[j] - true_onsets[i])
            if abs(difference) < tolerance:
                if i not in matches:
                    matches[i] = []
                matches[i].append((j, difference))

    # Select the best match for each detected onset
    for i in matches:
        matches[i] = sorted(matches[i], key=lambda x: abs(x[1]))
        matches[i] = matches[i][0]

    matches = [match for match in matches.values() if match is not None]

    true_positives = len(matches)
    assert true_positives <= len(true_onsets), "More true positives than true onsets"

    false_positives = len(detected_onsets) - true_positives
    false_negatives = len(true_onsets) - true_positives

    return OnsetEvaluationResult(
        true_positives=float(true_positives),
        false_positives=float(false_positives),
        false_negatives=float(false_negatives),
    )


def onset_samples_to_seconds(onset_samples, sample_rate):
    """Convert onset samples to seconds."""
    return [float(sample) / float(sample_rate) for sample in onset_samples]
