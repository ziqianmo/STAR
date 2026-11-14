"""Paper-specified LR behavior, contrastive scaling, and weighted optimization."""

from contextlib import nullcontext
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from star.loss.info_nce import InfoNCE
from star.training import TrainingConfig, build_scheduler
from star.trainer import train


class TrainingRecipeTests(unittest.TestCase):
    def test_cosine_schedule_warms_up_ten_percent_of_all_steps(self):
        config = TrainingConfig()
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=config.lr)
        scheduler, warmup_steps = build_scheduler(config, optimizer, train_steps=100)
        self.assertEqual(warmup_steps, 10)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        rates = {}
        for step in range(1, 101):
            optimizer.step()
            scheduler.step()
            if step in (5, 10, 55, 100):
                rates[step] = optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(rates[5], 0.0005)
        self.assertAlmostEqual(rates[10], 0.001)
        self.assertAlmostEqual(rates[55], 0.0005)
        self.assertAlmostEqual(rates[100], 0)

    def test_bidirectional_infonce_matches_closed_form_paper_sum(self):
        features = torch.eye(2)
        temperature_scale = torch.tensor(2.0, requires_grad=True)
        loss = InfoNCE(nn.CrossEntropyLoss(), device="cpu")(
            features, features, temperature_scale
        )
        # Each direction has logits [2, 0] for the matching pair.
        self.assertAlmostEqual(loss.item(), 2 * math.log1p(math.exp(-2)), places=6)
        loss.backward()
        self.assertAlmostEqual(
            temperature_scale.grad.item(), -2 / (1 + math.exp(2)), places=6
        )

    def test_trainer_update_uses_both_paper_loss_coefficients(self):
        class TinySTAR(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(0.2))
                self.logit_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
                self.logit_scale_blocks = nn.Parameter(
                    torch.tensor(0.0), requires_grad=False
                )

            def forward(self, query, sequence):
                def output(count):
                    logits = torch.stack((self.value, self.value * 0)).expand(count, -1)
                    alignment = self.value.expand(count, 1, 2)
                    features = (1 + self.value).expand(count, 2)
                    return alignment, [logits] * 3, None, features, None

                return output(len(query)), output(len(sequence))

        class IdentityScaler:
            def scale(self, value):
                return value

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        config = TrainingConfig(
            device="cpu", verbose=False, scheduler=None, clip_grad=None
        )
        self.assertEqual(config.vsam_loss_weight, 0.5)
        self.assertEqual(config.classification_loss_weight, 0.03)
        model = TinySTAR()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        labels = torch.zeros(2, dtype=torch.long)
        batches = [
            (torch.ones(2, 3, 1, 1), [torch.ones(2, 3, 1, 1)] * 5, ["a", "b"], labels)
        ]
        losses = {
            "infoNCE": lambda a, b, scale: a.mean(),
            "VSAM_loss": lambda a, b, scale: 2 * a.mean(),
        }
        # Exercise the weighted training path on CPU, without a CUDA requirement.
        with patch("star.trainer.autocast", return_value=nullcontext()):
            actual_loss = train(
                config, model, batches, losses, optimizer, scaler=IdentityScaler()
            )
        expected_loss = 1.2 + 0.5 * 0.4 + 0.03 * 4 * math.log1p(math.exp(-0.2))
        expected_parameter = 0.2 - 0.1 * (2 - 0.12 / (1 + math.exp(0.2)))
        self.assertAlmostEqual(actual_loss, expected_loss, places=6)
        self.assertAlmostEqual(model.value.item(), expected_parameter, places=6)


if __name__ == "__main__":
    unittest.main()
