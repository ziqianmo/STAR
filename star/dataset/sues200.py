"""SUES-200 datasets using the shared frame and sequence loading behavior.

Pass the prepared train/test folders for the selected altitude. Numeric natural
sorting keeps frames such as 2.jpg before 10.jpg without renaming image files.
"""

from .common import (
    ImageRetrievalDataset,
    ImageSequenceRetrievalDataset,
    ImageSequenceTrainDataset,
    get_data,
    get_transforms,
    natural_sort_key,
)

__all__ = [
    "SUES200DatasetTrainSequence",
    "SUES200DatasetEval",
    "SUES200DatasetEvalSequenceD2S",
    "get_data",
    "get_transforms",
    "natural_sort_key",
]


class SUES200DatasetTrainSequence(ImageSequenceTrainDataset):
    """Satellite/drone-sequence pairs with contiguous training class labels."""


class SUES200DatasetEval(ImageRetrievalDataset):
    """Single-image retrieval using the original SUES-200 scene IDs."""


class SUES200DatasetEvalSequenceD2S(ImageSequenceRetrievalDataset):
    """Consecutive drone-frame queries for drone-to-satellite retrieval."""
