"""Regression checks for STAR paper Eqs. (7)-(9) and DBPM pooling."""

import math
import unittest

import torch

from star.models.convnext.make_model import DBPM


def descriptors(values):
    return [
        torch.tensor([[[[value]], [[(i + 1) / 6]]]], dtype=torch.float64)
        for i, value in enumerate(values)
    ]


def reference_radius(rings):
    """Scalar, per-sample reference using the paper's one-based indices."""
    count = len(rings)
    scores = []
    for r in range(2, count):
        current = rings[r - 1]
        inner = [
            sum(f[k] for f in rings[: r - 1]) / (r - 1) for k in range(len(current))
        ]
        outer = [
            sum(f[k] for f in rings[r:]) / (count - r) for k in range(len(current))
        ]
        din = math.dist(current, inner)
        dout = math.dist(current, outer)
        scores.append((abs(din - dout), r))
    return max(scores, key=lambda score: score[0])[1]


class DBPMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_mean_features_before_distance_and_radius_mapping(self):
        # Eq. (7)-(9) selects radius 2; averaging pairwise distances selects 5.
        rings = descriptors([1, 5, 7, 6, 2, 1])
        radius = DBPM(block=2).select_boundary(rings, torch.arange(1, 7))
        self.assertEqual(radius.tolist(), [2])

    def test_innermost_ring_changes_boundary_independently_for_each_image(self):
        first = descriptors([0, 2, 3, 4, 5, 6])
        second = descriptors([100, 2, 3, 4, 5, 6])
        batched = [torch.cat((a, b), dim=0) for a, b in zip(first, second)]
        radius = DBPM(block=2).select_boundary(batched, torch.arange(1, 7))
        self.assertEqual(radius.tolist(), [5, 2])

    def test_forward_matches_region_slices_and_backpropagates(self):
        generator = torch.Generator().manual_seed(41)
        for pool in ("avg", "max"):
            for overlap in (False, True):
                with self.subTest(pool=pool, no_overlap=overlap):
                    # A rectangular map also checks masking outside the outer crop.
                    x = -torch.rand(
                        3, 4, 14, 16, generator=generator, dtype=torch.float64
                    )
                    x.requires_grad_()
                    module = DBPM(block=2, pool=pool)
                    rings, _ = module.get_ring_features(x)
                    vectors = torch.stack(rings, dim=1).flatten(2).detach().tolist()
                    radii = [reference_radius(sample) for sample in vectors]
                    expected = []
                    for sample, r in zip(x, radii):
                        foreground = sample[:, 7 - r : 7 + r, 8 - r : 8 + r]
                        background = sample[:, 1:13, 2:14].clone()
                        if overlap:
                            background[:, 6 - r : 6 + r, 6 - r : 6 + r] = 0
                        expected.append(
                            torch.cat(
                                (
                                    module.pooling(foreground.unsqueeze(0)),
                                    module.pooling(background.unsqueeze(0)),
                                ),
                                dim=2,
                            )
                        )
                    output = module(x, no_overlap=overlap)
                    torch.testing.assert_close(
                        output, torch.cat(expected), rtol=1e-12, atol=1e-12
                    )
                    self.assertEqual(tuple(output.shape), (3, 4, 2, 1))
                    for i in range(3):
                        torch.testing.assert_close(
                            output[i : i + 1], module(x[i : i + 1], no_overlap=overlap)
                        )
                    output.square().sum().backward()
                    self.assertTrue(torch.isfinite(x.grad).all().item())
                    self.assertGreater(x.grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision_matches_cpu(self):
        generator = torch.Generator().manual_seed(83)
        x = torch.rand(2, 8, 12, 12, generator=generator)
        module = DBPM(block=2)
        expected = module(x)
        actual = module.cuda()(x.cuda().half()).float().cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
