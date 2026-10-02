"""Check automatic evaluation's public options and single-pass score collection."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from leaf_hierarchy import evaluation
from leaf_hierarchy.engine import evaluate_loader
from leaf_hierarchy.evaluation import parse_args
from leaf_hierarchy.hierarchy import METHODS, TASKS, compare_predictions, decode_logits


TAXONOMY = {
    "class_mappings": {"family": {"F0": 0, "F1": 1},
                       "genus": {"G0": 0, "G1": 1, "G2": 2},
                       "species": {"S0": 0, "S1": 1, "S2": 2, "S3": 3}},
    "species_to_genus": {"S0": "G0", "S1": "G0", "S2": "G1", "S3": "G2"},
    "genus_to_family": {"G0": "F0", "G1": "F1", "G2": "F1"},
}


class EvaluationTests(unittest.TestCase):
    def test_incompatible_output_tasks_are_rejected_without_changing_previous_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "evaluation"
            destination.mkdir()
            previous_files = {
                "evaluation_config.json": json.dumps({"status": "complete", "tasks": list(TASKS)}).encode(),
                "comparison_metrics.json": b'{"independent": {"images": 3}}',
                "scores.npz": b"previous score data",
            }
            for name, content in previous_files.items():
                (destination / name).write_bytes(content)
            checkpoint = {"config": {}, "model_spec": {"tasks": ["species"]}}
            with (patch.object(evaluation, "load_checkpoint", return_value=(None, checkpoint)),
                  patch.object(evaluation, "configure_runtime"),
                  patch.object(evaluation, "load_manifest") as load_manifest,
                  patch.object(evaluation, "evaluate_loader") as evaluate_loader):
                with self.assertRaisesRegex(ValueError, "different tasks.*another --output-dir"):
                    evaluation.main(["--checkpoint", str(Path(temporary) / "best.pt"),
                                     "--output-dir", str(destination), "--device", "cpu"])
                load_manifest.assert_not_called()
                evaluate_loader.assert_not_called()
            self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()}, previous_files)

    def test_split_selection_keeps_test_default_and_needs_no_method_option(self):
        self.assertEqual(parse_args(["--checkpoint", "best.pt"]).split, "test")
        for split in ("validation", "test"):
            self.assertEqual(parse_args(["--checkpoint", "best.pt", "--split", split]).split, split)
        for extra in (["--split", "train"], ["--method", "all"]):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(["--checkpoint", "best.pt", *extra])

    def test_one_image_pass_supplies_losses_and_all_three_decodings(self):
        logits = {
            "family": torch.log(torch.tensor([[.1, .9], [.8, .2], [.2, .8]])),
            "genus": torch.log(torch.tensor([[.1, .8, .1], [.8, .1, .1], [.1, .1, .8]])),
            "species": torch.log(torch.tensor([[.45, .05, .4, .1], [.7, .1, .1, .1], [.1, .1, .1, .7]])),
        }
        truth = {"family": [1, 0, 1], "genus": [1, 0, 2], "species": [2, 0, 3]}
        samples = [(torch.tensor(index), {task: truth[task][index] for task in TASKS}) for index in range(3)]

        class CountedPredictions(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.seen = []

            def forward(self, images):
                self.calls += 1
                self.seen.extend(images.tolist())
                return {task: logits[task][images] for task in TASKS}

        model = CountedPredictions()
        loader = DataLoader(samples, batch_size=2)
        with redirect_stdout(StringIO()):
            result = evaluate_loader(model, loader, torch.device("cpu"), tasks=TASKS,
                                     class_mappings=TAXONOMY["class_mappings"], taxonomy=TAXONOMY,
                                     consistency_weight=.7, collect_scores=True, progress=True)
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.seen, [0, 1, 2])
        self.assertEqual(result["true_labels"], truth)
        for task in TASKS:
            np.testing.assert_array_equal(result["logits"][task], logits[task].numpy())
            np.testing.assert_array_equal(result["predicted_labels"][task], logits[task].argmax(1))
        self.assertAlmostEqual(result["weighted_consistency_loss"], .7 * result["consistency_loss"], places=6)
        self.assertAlmostEqual(result["loss"], result["classification_loss"]
                               + result["weighted_consistency_loss"], places=6)
        predictions, _ = decode_logits(result["logits"], TAXONOMY)
        metrics = compare_predictions(result["true_labels"], predictions, TAXONOMY)
        self.assertEqual(set(metrics), set(METHODS))
        self.assertEqual(metrics["independent"]["invalid_paths"], 1)
        self.assertEqual(metrics["species_path"]["coherence_rate"], 1.)
        self.assertEqual(metrics["joint_path"]["coherence_rate"], 1.)
        self.assertEqual(metrics["joint_path"]["tasks"]["species"]["fixed_from_independent"], 1)
        for task in TASKS:
            for key in ("accuracy", "macro_f1"):
                self.assertEqual(result["tasks"][task][key], metrics["independent"]["tasks"][task][key])

        # Training's evaluation path remains lightweight and independent by default.
        ordinary = evaluate_loader(CountedPredictions(), loader, torch.device("cpu"), tasks=TASKS,
                                   class_mappings=TAXONOMY["class_mappings"], taxonomy=TAXONOMY,
                                   consistency_weight=.7)
        self.assertNotIn("logits", ordinary)
        self.assertNotIn("methods", ordinary)
        for key in ("loss", "tasks", "predicted_labels", "hierarchy", "consistency_loss"):
            self.assertEqual(ordinary[key], result[key])


if __name__ == "__main__":
    unittest.main()
