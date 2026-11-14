"""University-1652 dataset entry points with the existing public API."""

from .common import (
    ImagePairTrainDataset,
    ImageRetrievalDataset,
    ImageSequenceRetrievalDataset,
    ImageSequenceTrainDataset,
    get_data,
    get_transforms,
    natural_sort_key,
)

__all__ = [
    "U1652DatasetTrain",
    "U1652DatasetEval",
    "U1652DatasetTrainSequence",
    "U1652DatasetEvalSequenceD2S",
    "get_data",
    "get_transforms",
    "natural_sort_key",
]


class U1652DatasetTrain(ImagePairTrainDataset):
    """Single-image satellite/drone training pairs."""


class U1652DatasetEval(ImageRetrievalDataset):
    """Single-image retrieval using University-1652 building IDs."""


class U1652DatasetTrainSequence(ImageSequenceTrainDataset):
    """Satellite/drone-sequence pairs with contiguous training class labels."""


class U1652DatasetEvalSequenceD2S(ImageSequenceRetrievalDataset):
    """Consecutive drone-frame queries for drone-to-satellite retrieval."""
