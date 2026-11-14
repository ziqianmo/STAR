"""Configuration remains editable from Python and portable across working directories."""

from dataclasses import replace
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from star.training import TrainingConfig, configure as configure_training
from star.evaluate.cli import EvaluationConfig, configure as configure_evaluation


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for prefix in ("", "300/"):
            for path in (
                "train/" + prefix + "satellite",
                "train/" + prefix + "drone",
                "test/" + prefix + "query_drone",
                "test/" + prefix + "gallery_satellite",
                "test/" + prefix + "query_satellite",
                "test/" + prefix + "gallery_drone",
            ):
                (self.root / path).mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.root / "weights.pth"
        self.checkpoint.touch()

    def test_entry_main_uses_visible_config(self):
        for name in (
            "train_university",
            "train_sues200",
            "eval_university",
            "eval_university_seq_D2S",
            "eval_sues200",
            "eval_sues200_seq_D2S",
        ):
            with self.subTest(name=name):
                entry = importlib.import_module(name)
                runner = (
                    "run_training" if name.startswith("train") else "run_evaluation"
                )
                sentinel = object()
                with patch.object(entry, "config", sentinel), patch.object(
                    entry, runner
                ) as run:
                    entry.main(["--help"])
                    self.assertIs(run.call_args.args[0], sentinel)

    def test_training_override_does_not_mutate_defaults(self):
        default = TrainingConfig(
            data_folder=str(self.root), batch_size=6, lr=0.002, device="cpu"
        )
        actual = configure_training(default, ["--batch-size", "4", "--lr", "0.003"])
        self.assertEqual((actual.batch_size, actual.lr), (4, 0.003))
        self.assertEqual((default.batch_size, default.lr), (6, 0.002))
        self.assertFalse(hasattr(default, "query_folder_train"))
        self.assertEqual(Path(actual.query_folder_test), self.root / "test/query_drone")

    def test_python_evaluation_limits_are_preserved(self):
        default = EvaluationConfig(
            data_folder=str(self.root),
            checkpoint_start=str(self.checkpoint),
            device="cpu",
            max_queries=7,
            max_gallery=11,
            verbose=False,
            output=str(self.root / "results.json"),
            benchmark_iterations=3,
        )
        actual = configure_evaluation(default, [])
        self.assertEqual(
            (actual.max_queries, actual.max_gallery, actual.benchmark_iterations),
            (7, 11, 3),
        )
        self.assertEqual(actual.output, default.output)
        self.assertFalse(actual.verbose)
        changed = configure_evaluation(default, ["--max-queries", "2"])
        self.assertEqual(changed.max_queries, 2)
        self.assertEqual(default.max_queries, 7)

    def test_sues_paths_and_old_university_direction_alias(self):
        config = TrainingConfig(
            dataset_name="sues200", data_folder=str(self.root), device="cpu"
        )
        actual = configure_training(config, ["--direction", "S2D"])
        self.assertEqual(
            Path(actual.query_folder_train), self.root / "train/300/satellite"
        )
        self.assertEqual(
            Path(actual.query_folder_test), self.root / "test/300/query_satellite"
        )
        config = TrainingConfig(data_folder=str(self.root), device="cpu")
        actual = configure_training(config, ["--dataset", "U1652-S2D"])
        self.assertEqual(actual.direction, "S2D")

    def test_project_defaults_are_absolute(self):
        for config in (TrainingConfig(), EvaluationConfig()):
            self.assertTrue(Path(config.data_folder).is_absolute())
        self.assertTrue(Path(TrainingConfig().model_path).is_absolute())

    def test_experiment_defaults_are_retained(self):
        university = importlib.import_module("train_university").Configuration()
        sues = importlib.import_module("train_sues200").Configuration()
        self.assertEqual(
            (university.lr, university.pretrained, university.nclasses),
            (0.001, False, 701),
        )
        self.assertEqual((sues.lr, sues.pretrained, sues.nclasses), (0.0006, True, 200))


if __name__ == "__main__":
    unittest.main()
