"""Dataset entry points, numeric frame order, and trainer return contracts."""

import contextlib
import io
from pathlib import Path
import pickle
import random
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from star.dataset.sues200 import (
    SUES200DatasetEval,
    SUES200DatasetEvalSequenceD2S,
    SUES200DatasetTrainSequence,
)
from star.dataset.university import U1652DatasetTrain, U1652DatasetTrainSequence


class DatasetEntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for dataset, padded in (("sues", False), ("university", True)):
            for scene in ("0007", "0021"):
                satellite = self.root / dataset / "satellite" / scene
                drone = self.root / dataset / "drone" / scene
                satellite.mkdir(parents=True)
                drone.mkdir(parents=True)
                self.assertTrue(
                    cv2.imwrite(
                        str(satellite / "0.png"), np.full((16, 16, 3), 127, np.uint8)
                    )
                )
                # Create files out of order so enumeration order is not a fallback.
                for index in reversed(range(12)):
                    name = f"image-{index+1:02d}.jpeg" if padded else f"{index}.jpg"
                    self.assertTrue(
                        cv2.imwrite(
                            str(drone / name), np.full((16, 16, 3), index, np.uint8)
                        )
                    )

    def tearDown(self):
        self.temporary.cleanup()

    def training(self, dataset="sues"):
        cls = (
            SUES200DatasetTrainSequence
            if dataset == "sues"
            else U1652DatasetTrainSequence
        )
        return cls(
            str(self.root / dataset / "satellite"),
            str(self.root / dataset / "drone"),
            prob_flip=0,
            shuffle_batch_size=2,
            sequence_length=5,
        )

    def test_sues_frames_are_numeric_and_sequences_stay_within_each_scene(self):
        dataset = self.training()
        self.assertEqual(len(dataset), 16)
        self.assertEqual(
            [Path(p).name for p in dataset.pairs[0][3]],
            ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg"],
        )
        for scene, label, satellite, frames in dataset.pairs:
            numbers = [int(Path(p).stem) for p in frames]
            self.assertEqual(numbers, list(range(numbers[0], numbers[0] + 5)))
            self.assertEqual(Path(satellite).parent.name, scene)
            self.assertTrue(all(Path(p).parent.name == scene for p in frames))
            self.assertEqual(label, {"0007": 0, "0021": 1}[scene])

    def test_university_padded_order_and_existing_return_formats_are_preserved(self):
        dataset = self.training("university")
        self.assertEqual(
            [Path(p).name for p in dataset.pairs[0][3]],
            [f"image-{i:02d}.jpeg" for i in range(1, 6)],
        )
        self.assertEqual(len(dataset[0]), 4)
        single = U1652DatasetTrain(
            str(self.root / "university/satellite"),
            str(self.root / "university/drone"),
            prob_flip=0,
        )
        self.assertEqual(len(single), 24)
        self.assertEqual(len(single[0]), 3)

    def test_single_and_sequence_retrieval_preserve_scene_ids_and_junk_labels(self):
        folder = str(self.root / "sues/drone")
        single = SUES200DatasetEval(folder, mode="query", sample_ids={"0007"})
        sequence = SUES200DatasetEvalSequenceD2S(
            folder, mode="query", sample_ids={"0007"}
        )
        self.assertEqual(len(single), 24)
        self.assertEqual(len(sequence), 16)
        self.assertEqual(sequence.get_sample_ids(), {"0007", "0021"})
        for i, scene in enumerate(single.sample_ids):
            self.assertEqual(single[i][1], 7 if scene == "0007" else -1)
        for i, scene in enumerate(sequence.sample_ids):
            frames, label = sequence[i]
            self.assertEqual(len(frames), 5)
            self.assertEqual(label, 7 if scene == "0007" else -1)

    def test_sues_dataset_pickles_and_collates_for_the_existing_trainer(self):
        dataset = pickle.loads(pickle.dumps(self.training()))
        self.assertIsInstance(dataset, SUES200DatasetTrainSequence)
        loader = DataLoader(Subset(dataset, [0, 8]), batch_size=2)
        query, frames, scene_ids, labels = next(iter(loader))
        self.assertEqual(tuple(query.shape), (2, 16, 16, 3))
        self.assertEqual(len(frames), 5)
        self.assertEqual(tuple(frames[0].shape), (2, 16, 16, 3))
        self.assertEqual(tuple(scene_ids), ("0007", "0021"))
        self.assertEqual(labels.tolist(), [0, 1])

    def test_custom_sampling_keeps_distinct_scenes_and_frame_order(self):
        dataset = self.training()
        random.seed(21)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            dataset.shuffle()
        self.assertEqual(len(dataset), 16)
        for i in range(0, len(dataset), 2):
            self.assertEqual(len({pair[0] for pair in dataset.samples[i : i + 2]}), 2)
        for _, _, _, paths in dataset.samples:
            numbers = [int(Path(p).stem) for p in paths]
            self.assertEqual(numbers, list(range(numbers[0], numbers[0] + 5)))


if __name__ == "__main__":
    unittest.main()
