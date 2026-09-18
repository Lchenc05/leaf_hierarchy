"""Exercise multitask gradients, independent metrics and saved-model workflows."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from leaf_hierarchy import evaluation, prediction, training
from leaf_hierarchy.checkpoints import load_checkpoint
from leaf_hierarchy.config import load_config
from leaf_hierarchy.data import make_loader, load_manifest
from leaf_hierarchy.engine import evaluate_loader, task_loss
from leaf_hierarchy.models import create_model, validate_model_spec
from leaf_hierarchy.runtime import ROOT, configure_runtime

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader


TASKS = ["family", "genus", "species"]
MAPPINGS = {"family": {"F": 0, "G": 1}, "genus": {"A": 0, "B": 1, "C": 2},
            "species": {"A a": 0, "A b": 1, "B a": 2, "C a": 3}}
SPEC = {"architecture": "resnet18", "tasks": TASKS}


class MultitaskTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(num_threads=2)

    def test_shapes_gradients_and_baseline_initialization(self):
        torch.manual_seed(17)
        baseline = create_model({"architecture": "resnet18", "tasks": ["species"]}, MAPPINGS)
        torch.manual_seed(17)
        model = create_model(SPEC, MAPPINGS)
        self.assertTrue(torch.equal(baseline.backbone.conv1.weight, model.backbone.conv1.weight))
        self.assertTrue(torch.equal(baseline.backbone.fc.weight, model.heads["species"].weight))
        images = torch.randn(2, 3, 64, 64)
        targets = {task: torch.tensor([0, 1]) for task in TASKS}
        outputs = model(images)
        self.assertEqual(list(outputs), TASKS)
        for task in TASKS:
            self.assertEqual(tuple(outputs[task].shape), (2, len(MAPPINGS[task])))
            gradient = torch.autograd.grad(nn.functional.cross_entropy(outputs[task], targets[task]),
                                          model.backbone.conv1.weight, retain_graph=True)[0]
            self.assertGreater(gradient.abs().sum().item(), 0)
        loss = task_loss(outputs, targets, TASKS)
        expected = sum(nn.functional.cross_entropy(outputs[t], targets[t]) for t in TASKS)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for task in TASKS:
            self.assertGreater(model.heads[task].weight.grad.abs().sum().item(), 0)

    def test_model_rejects_missing_mappings_and_unsupported_tasks(self):
        for tasks in ([], ["family"], ["species", "species"], ["species", "genus", "family"]):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                validate_model_spec({"architecture": "resnet18", "tasks": tasks})
        for task in TASKS:
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, task):
                create_model(SPEC, {key: value for key, value in MAPPINGS.items() if key != task})

    def test_experiment_matches_baseline_comparison_settings(self):
        baseline = load_config(ROOT / "experiments/resnet18_species/config.toml")
        multitask = load_config(ROOT / "experiments/resnet18_multitask/config.toml")
        for key in ("data", "training", "runtime", "selection", "preprocessing", "augmentation"):
            self.assertEqual(baseline[key], multitask[key])
        self.assertEqual(multitask["model"]["tasks"], TASKS)

    def test_metrics_and_losses_are_independent_with_uneven_batches(self):
        class Predictions(nn.Module):
            def forward(self, images):
                return {task: images[:, index] for index, task in enumerate(TASKS)}

        logits = torch.tensor([[[4., 0.], [4., 0.], [0., 4.]],
                               [[4., 0.], [0., 4.], [0., 4.]],
                               [[4., 0.], [4., 0.], [0., 4.]]])
        samples = [(row, {task: 0 for task in TASKS}) for row in logits]
        result = evaluate_loader(Predictions(), DataLoader(samples, batch_size=2), torch.device("cpu"),
                                 tasks=TASKS, class_mappings={task: {"a": 0, "b": 1} for task in TASKS})
        for index, (accuracy, macro_f1) in enumerate(((1., .5), (2/3, .4), (0., 0.))):
            metrics = result["tasks"][TASKS[index]]
            self.assertAlmostEqual(metrics["accuracy"], accuracy)
            self.assertAlmostEqual(metrics["macro_f1"], macro_f1)
            expected = nn.functional.cross_entropy(logits[:, index], torch.zeros(3, dtype=torch.long))
            self.assertAlmostEqual(metrics["loss"], expected.item(), places=6)
        self.assertAlmostEqual(result["loss"], sum(m["loss"] for m in result["tasks"].values()), places=6)

    def test_train_reload_evaluate_and_predict_for_both_experiments(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ), redirect_stdout(StringIO()):
            os.environ.pop("LEAF_HIERARCHY_DATA_DIR", None)
            root = Path(temporary)
            records = []
            for index, (species, genus, family) in enumerate(
                    (("A a", "A", "F"), ("A b", "A", "F"), ("B a", "B", "F"), ("C a", "C", "G"))):
                for split_index, split in enumerate(("train", "validation", "test")):
                    name = f"{index}_{split}.png"
                    Image.new("RGB", (64, 64), (index * 60, split_index * 70, 30)).save(root / name)
                    records.append({"image_path": name, "species": species, "genus": genus, "family": family,
                                    "class_id": str(index), "observation_id": name, "group_id": name, "split": split})
            manifest = root / "split.csv"
            pd.DataFrame(records).to_csv(manifest, index=False)
            for tasks in (["species"], TASKS):
                with self.subTest(tasks=tasks):
                    config_file = root / "config.toml"
                    config_file.write_text(
                        f'[model]\ntasks = {json.dumps(tasks)}\nweights = "none"\n'
                        '[training]\nepochs = 2\nbatch_size = 4\n'
                        '[preprocessing]\nresize_size = 64\ncrop_size = 64\n'
                        '[runtime]\ndevice = "cpu"\nnum_threads = 2\n', encoding="utf-8")
                    output = root / f"runs_{len(tasks)}"
                    training.main(["--config", str(config_file), "--data-dir", str(root),
                                   "--split-file", str(manifest), "--output-dir", str(output)])
                    run = next(output.iterdir())
                    self.assertFalse((run / "test").exists())
                    model, checkpoint = load_checkpoint(run / "best.pt", torch.device("cpu"))
                    history = pd.read_csv(run / "history.csv")
                    best = history.sort_values(["val_species_macro_f1", "val_loss"],
                                               ascending=[False, True], kind="stable").iloc[0]
                    self.assertEqual(checkpoint["epoch"], best["epoch"])
                    frame, _ = load_manifest(manifest, root)
                    loader = make_loader(frame, "validation", root, 4, class_mappings=MAPPINGS,
                                         tasks=tasks, preprocessing=checkpoint["preprocessing"])
                    restored = evaluate_loader(model, loader, torch.device("cpu"), tasks=tasks,
                                               class_mappings=MAPPINGS)
                    evaluation.main(["--checkpoint", str(run / "best.pt"), "--device", "cpu", "--num-threads", "2"])
                    for split in ("validation", "test"):
                        combined = json.loads((run / split / f"{split}_metrics.json").read_text())
                        self.assertEqual(combined["selected_epoch"], checkpoint["epoch"])
                        self.assertEqual(set(combined["tasks"]), set(tasks))
                        predictions = pd.read_csv(run / split / f"{split}_predictions.csv")
                        self.assertEqual(predictions["image_path"].tolist(),
                                         frame.loc[frame["split"] == split, "image_path"].tolist())
                        for task in tasks:
                            metrics = json.loads((run / split / f"{split}_{task}_metrics.json").read_text())
                            for metric in ("accuracy", "macro_f1", "loss", "num_classes"):
                                self.assertEqual(metrics[metric], combined["tasks"][task][metric])
                                if split == "validation":
                                    self.assertAlmostEqual(metrics[metric], restored["tasks"][task][metric])
                            matrix = pd.read_csv(run / split / f"{split}_{task}_confusion_matrix.csv", index_col=0)
                            self.assertEqual(matrix.shape, (len(MAPPINGS[task]), len(MAPPINGS[task])))
                            self.assertEqual(matrix.values.sum(), 4)
                            self.assertTrue((run / split / f"{split}_{task}_classification_report.csv").is_file())
                            self.assertIn(f"val_{task}_loss", history)
                    output_json = StringIO()
                    with redirect_stdout(output_json):
                        prediction.main(["--checkpoint", str(run / "best.pt"), "--image", str(root / "0_test.png"),
                                         "--device", "cpu", "--num-threads", "2", "--json"])
                    self.assertEqual(set(json.loads(output_json.getvalue())["predictions"]), set(tasks))


if __name__ == "__main__":
    unittest.main()
