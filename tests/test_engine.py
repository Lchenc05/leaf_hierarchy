"""Check task alignment and metric semantics independently of an image backbone."""

import unittest

from leaf_hierarchy import runtime  # Configure deterministic CUDA before torch.
from leaf_hierarchy.engine import evaluate_loader, task_loss

import torch
from torch import nn
from torch.utils.data import DataLoader


class EchoModel(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, images):
        return {self.task: images}


class EngineTests(unittest.TestCase):
    def test_single_task_loss_and_gradient_match_cross_entropy(self):
        for task in ("species", "genus", "family"):
            with self.subTest(task=task):
                logits = torch.tensor([[1.0, -0.5], [-2.0, 0.2]], requires_grad=True)
                targets = torch.tensor([0, 1])
                expected = nn.CrossEntropyLoss()(logits, targets)
                actual = task_loss({task: logits}, {task: targets}, [task])
                self.assertTrue(torch.equal(expected, actual))
                expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
                self.assertTrue(torch.equal(expected_gradient, torch.autograd.grad(actual, logits)[0]))

    def test_task_mismatch_fails(self):
        logits = torch.zeros(2, 2)
        targets = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "outputs"):
            task_loss({"family": logits}, {"species": targets}, ["species"])
        with self.assertRaisesRegex(ValueError, "Targets"):
            task_loss({"species": logits}, {"family": targets}, ["species"])

    def test_metrics_keep_missing_classes_in_macro_average_and_sample_order(self):
        for task in ("species", "genus", "family"):
            with self.subTest(task=task):
                samples = [(torch.tensor([2.0, 0.0, -1.0]), {task: 0}),
                           (torch.tensor([0.0, 2.0, -1.0]), {task: 1}),
                           (torch.tensor([0.0, 2.0, -1.0]), {task: 0})]
                result = evaluate_loader(
                    EchoModel(task), DataLoader(samples, batch_size=2), torch.device("cpu"),
                    tasks=[task], class_mappings={task: {"a": 0, "b": 1, "c": 2}},
                )
                self.assertEqual(result["true_labels"][task], [0, 1, 0])
                self.assertEqual(result["predicted_labels"][task], [0, 1, 1])
                self.assertAlmostEqual(result["tasks"][task]["accuracy"], 2 / 3)
                self.assertAlmostEqual(result["tasks"][task]["macro_f1"], 4 / 9)
                all_logits = torch.stack([sample[0] for sample in samples])
                expected_loss = nn.CrossEntropyLoss()(all_logits, torch.tensor([0, 1, 0])).item()
                self.assertAlmostEqual(result["loss"], expected_loss, places=6)


if __name__ == "__main__":
    unittest.main()
