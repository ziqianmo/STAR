"""Retrieval metrics, sequence descriptors, checkpoint loading, and CLI paths."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn

from star.evaluate.checkpoints import load_checkpoint
from star.evaluate.cli import EvaluationConfig, configure
from star.evaluate.features import encode_images, extract_features
from star.evaluate.metrics import compute_mAP, compute_metrics


class FlattenModel(nn.Module):
    def forward(self, images):
        return images.flatten(start_dim=1)


class TinySTAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_1 = nn.Module()
        self.model_1.convnext = nn.Linear(2, 2)
        self.model_1.classifier = nn.Linear(2, 3)


class RetrievalTests(unittest.TestCase):
    def test_original_trapezoidal_ap_and_junk_removal(self):
        ap, cmc = compute_mAP([0, 1, 2, 3], np.array([[0], [2]]), [])
        self.assertAlmostEqual(ap, 19 / 24)
        self.assertEqual(cmc.tolist(), [1, 1, 1, 1])
        ap, cmc = compute_mAP([0, 1, 2], [2], [0])
        self.assertAlmostEqual(ap, 0.25)
        self.assertEqual(cmc.tolist(), [0, 1, 1])

    def test_top_one_percent_uses_the_correct_boundary_rank(self):
        gallery = torch.arange(200, 0, -1).float().reshape(200, 1)
        for positive, expected in ((1, 1.0), (2, 0.0)):
            with self.subTest(positive=positive):
                metrics = compute_metrics(
                    torch.ones(1, 1),
                    [positive],
                    gallery,
                    torch.arange(200),
                    verbose=False,
                )
                self.assertEqual(metrics["top_1_percent_rank"], 2)
                self.assertEqual(metrics["recall_at_1_percent"], expected)

    def test_singleton_gallery_supports_all_requested_ranks(self):
        metrics = compute_metrics(
            torch.ones(1, 1), [7], torch.ones(1, 1), [7], verbose=False
        )
        self.assertEqual(metrics["recall_at_k"], {"1": 1.0, "5": 1.0, "10": 1.0})
        self.assertEqual(metrics["top_1_percent_rank"], 1)
        self.assertEqual(metrics["mAP"], 1)

    def test_chunking_and_queries_without_positives(self):
        query = torch.eye(3)
        gallery = torch.eye(3)[:2]
        first = compute_metrics(
            query, [1, 2, 3], gallery, [1, 2], step_size=1, verbose=False
        )
        second = compute_metrics(
            query, [1, 2, 3], gallery, [1, 2], step_size=3, verbose=False
        )
        self.assertEqual(first, second)
        self.assertEqual(first["queries_without_positive"], 1)
        self.assertAlmostEqual(first["recall_at_k"]["1"], 2 / 3)
        self.assertAlmostEqual(first["mAP"], 2 / 3)

    def test_empty_data_and_nonfinite_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "contain samples"):
            compute_metrics(torch.empty(0, 2), [], torch.ones(1, 2), [1], verbose=False)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compute_metrics(
                torch.tensor([[float("nan")]]),
                [1],
                torch.ones(1, 1),
                [1],
                verbose=False,
            )

    def test_sequence_lists_and_stacked_tensors_mean_before_normalizing(self):
        config = SimpleNamespace(
            device="cpu",
            handcraft_model=False,
            mixed_precision=True,
            normalize_features=True,
            verbose=False,
        )
        frames = torch.tensor(
            [[[3.0, 0.0], [0.0, 1.0]], [[0.0, 2.0], [4.0, 0.0]]]
        ).reshape(2, 2, 2, 1, 1)
        tensor_output = encode_images(config, FlattenModel(), frames, sequence=True)
        list_output = encode_images(
            config, FlattenModel(), [frames[:, 0], frames[:, 1]], sequence=True
        )
        expected = torch.nn.functional.normalize(frames.mean(dim=1).flatten(1), dim=-1)
        torch.testing.assert_close(tensor_output, expected)
        torch.testing.assert_close(list_output, expected)
        features, ids = extract_features(
            config, FlattenModel(), [(frames, torch.tensor([7, 9]))], sequence=True
        )
        torch.testing.assert_close(features, expected)
        self.assertEqual(ids.tolist(), [7, 9])


class CheckpointTests(unittest.TestCase):
    def test_nested_dataparallel_checkpoint_and_different_classifier(self):
        model = TinySTAR()
        state = {
            f"module.{key}": torch.full_like(value, 0.25)
            for key, value in model.state_dict().items()
        }
        state["module.model_1.classifier.weight"] = torch.zeros(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "weights.pth"
            torch.save({"state_dict": state, "epoch": 1}, filename)
            report = load_checkpoint(model, filename)
        torch.testing.assert_close(
            model.model_1.convnext.weight, torch.full((2, 2), 0.25)
        )
        self.assertEqual(report["backbone_keys"], 2)
        self.assertIn("model_1.classifier.weight", report["skipped_checkpoint_keys"])

    def test_incomplete_backbone_fails_instead_of_evaluating_random_weights(self):
        model = TinySTAR()
        state = model.state_dict()
        del state["model_1.convnext.weight"]
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "weights.pth"
            torch.save(state, filename)
            with self.assertRaisesRegex(ValueError, "backbone weights"):
                load_checkpoint(model, filename)


class EvaluationConfigurationTests(unittest.TestCase):
    def test_explicit_paths_direction_and_altitude(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for view in ("query_satellite", "gallery_drone"):
                (root / "test/250" / view).mkdir(parents=True)
            checkpoint = root / "weights.pth"
            checkpoint.touch()
            config = configure(
                EvaluationConfig(dataset_name="sues200", device="cpu"),
                [
                    "--data-folder",
                    str(root),
                    "--checkpoint",
                    str(checkpoint),
                    "--direction",
                    "S2D",
                    "--altitude",
                    "250",
                    "--num-workers",
                    "0",
                ],
            )
            self.assertEqual(
                config.query_folder_test, str(root / "test/250/query_satellite")
            )
            self.assertEqual(
                config.gallery_folder_test, str(root / "test/250/gallery_drone")
            )
            self.assertFalse(config.pretrained)
            self.assertEqual(config.gpu_ids, ())

    def test_sequence_evaluation_rejects_the_s2d_direction(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                configure(
                    EvaluationConfig(sequence_query=True, device="cpu"),
                    ["--checkpoint", "unused.pth", "--direction", "S2D"],
                )
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
