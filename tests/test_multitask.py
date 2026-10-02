"""Exercise multitask gradients, independent metrics and saved-model workflows."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from leaf_hierarchy import evaluation, hierarchy_evaluation, prediction, training
from leaf_hierarchy.checkpoints import load_checkpoint
from leaf_hierarchy.config import load_config
from leaf_hierarchy.data import make_loader, load_manifest
from leaf_hierarchy.engine import evaluate_loader, task_loss
from leaf_hierarchy.hierarchy import METHODS
from leaf_hierarchy.models import create_model, validate_model_spec
from leaf_hierarchy.runtime import ROOT, configure_runtime

import pandas as pd
import numpy as np
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
        regularized = load_config(ROOT / "experiments/resnet18_consistency/config.toml")
        for key in ("data", "runtime", "selection", "preprocessing", "augmentation"):
            self.assertEqual(baseline[key], multitask[key])
            self.assertEqual(multitask[key], regularized[key])
        # Seeds are per-run choices and can differ in the user's editable configurations.
        for key, value in baseline["training"].items():
            if key != "seed":
                self.assertEqual(value, multitask["training"][key])
            if key not in ("seed", "consistency_weight"):
                self.assertEqual(value, regularized["training"][key])
        self.assertEqual(multitask["model"]["tasks"], TASKS)
        self.assertEqual(regularized["model"], multitask["model"])
        self.assertEqual(regularized["training"]["consistency_weight"], 1.0)

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
            multitask_runs = []
            for tasks, consistency_weight in ((["species"], 0.0), (TASKS, 0.0), (TASKS, 0.4)):
                with self.subTest(tasks=tasks, consistency_weight=consistency_weight):
                    config_file = root / "config.toml"
                    config_file.write_text(
                        f'[model]\ntasks = {json.dumps(tasks)}\nweights = "none"\n'
                        f'[training]\nepochs = 2\nbatch_size = 4\nconsistency_weight = {consistency_weight}\n'
                        '[preprocessing]\nresize_size = 64\ncrop_size = 64\n'
                        '[runtime]\ndevice = "cpu"\nnum_threads = 2\n', encoding="utf-8")
                    output = root / f"runs_{len(tasks)}_{consistency_weight}"
                    training.main(["--config", str(config_file), "--data-dir", str(root),
                                   "--split-file", str(manifest), "--output-dir", str(output)])
                    run = next(output.iterdir())
                    self.assertFalse((run / "test").exists())
                    training_metrics = json.loads((run / "validation/validation_metrics.json").read_text())
                    self.assertNotIn("methods", training_metrics)
                    self.assertFalse((run / "validation/comparison_metrics.json").exists())
                    model, checkpoint = load_checkpoint(run / "best.pt", torch.device("cpu"))
                    self.assertEqual(checkpoint["config"]["training"]["consistency_weight"], consistency_weight)
                    history = pd.read_csv(run / "history.csv")
                    best = history.sort_values(["val_species_macro_f1", "val_loss"],
                                               ascending=[False, True], kind="stable").iloc[0]
                    self.assertEqual(checkpoint["epoch"], best["epoch"])
                    frame, _ = load_manifest(manifest, root)
                    loader = make_loader(frame, "validation", root, 4, class_mappings=MAPPINGS,
                                         tasks=tasks, preprocessing=checkpoint["preprocessing"])
                    restored = evaluate_loader(model, loader, torch.device("cpu"), tasks=tasks,
                                               class_mappings=MAPPINGS, taxonomy=checkpoint["taxonomy"],
                                               consistency_weight=consistency_weight)
                    evaluate_args = ["--checkpoint", str(run / "best.pt"), "--device", "cpu", "--num-threads", "2"]
                    if len(tasks) == 3:
                        # A runtime/data TOML must not replace the checkpoint's trained objective.
                        runtime_config = root / "evaluate.toml"
                        runtime_config.write_text(f'[model]\ntasks = {json.dumps(tasks)}\n'
                                                  '[training]\nconsistency_weight = 0.9\n', encoding="utf-8")
                        evaluate_args += ["--config", str(runtime_config), "--data-dir", str(root),
                                          "--split-file", str(manifest)]
                    # Validation must not load any training/test image or create test results.
                    original_open = Image.open
                    opened_images = []

                    def validation_images_only(path, *args, **kwargs):
                        self.assertTrue(Path(path).name.endswith("_validation.png"), str(path))
                        opened_images.append(Path(path).name)
                        return original_open(path, *args, **kwargs)

                    with patch("PIL.Image.open", side_effect=validation_images_only):
                        evaluation.main(evaluate_args + ["--split", "validation"])
                    self.assertEqual(len(opened_images), 4)
                    self.assertFalse((run / "test").exists())
                    evaluation.main(evaluate_args)
                    if len(tasks) == 3:
                        # Re-evaluation must replace the same results without directory errors.
                        evaluation.main(evaluate_args)
                        multitask_runs.append(run)
                    for split in ("validation", "test"):
                        combined = json.loads((run / split / f"{split}_metrics.json").read_text())
                        self.assertEqual(combined["selected_epoch"], checkpoint["epoch"])
                        self.assertEqual(set(combined["tasks"]), set(tasks))
                        metadata = json.loads((run / split / "evaluation_config.json").read_text())
                        self.assertEqual(metadata["split"], split)
                        if split == "validation":
                            for key, value in training_metrics.items():
                                self.assertEqual(combined[key], value)
                        predictions = pd.read_csv(run / split / f"{split}_predictions.csv")
                        self.assertEqual(predictions["image_path"].tolist(),
                                         frame.loc[frame["split"] == split, "image_path"].tolist())
                        if len(tasks) == 3:
                            compared = json.loads((run / split / "comparison_metrics.json").read_text())
                            self.assertEqual(combined["methods"], compared)
                            self.assertEqual(set(compared), set(METHODS))
                            summary = pd.read_csv(run / split / "summary.csv")
                            self.assertEqual(len(summary), 9)
                            self.assertEqual(set(summary.method), set(METHODS))
                            self.assertEqual(set(summary.split), {split})
                            for method in METHODS:
                                method_dir = run / split / method
                                self.assertEqual(json.loads((method_dir / "metrics.json").read_text()), compared[method])
                                for task in tasks:
                                    matrix = pd.read_csv(method_dir / f"{task}_confusion_matrix.csv", index_col=0)
                                    self.assertEqual(matrix.values.sum(), 4)
                                    self.assertTrue((method_dir / f"{task}_classification_report.csv").is_file())
                                    self.assertEqual(compared["independent"]["tasks"][task]["accuracy"],
                                                     combined["tasks"][task]["accuracy"])
                                    self.assertEqual(compared["independent"]["tasks"][task]["macro_f1"],
                                                     combined["tasks"][task]["macro_f1"])
                            self.assertEqual(combined[f"{split}_consistency_weight"], consistency_weight)
                            self.assertAlmostEqual(combined[f"{split}_weighted_consistency_loss"],
                                                   consistency_weight * combined[f"{split}_consistency_loss"], places=6)
                            self.assertAlmostEqual(combined[f"{split}_loss"],
                                                   combined[f"{split}_classification_loss"]
                                                   + combined[f"{split}_weighted_consistency_loss"], places=6)
                            self.assertAlmostEqual(combined["hierarchy"]["coherence_rate"], predictions.valid_path.mean())
                            self.assertEqual(combined["hierarchy"]["invalid_paths"], int((~predictions.valid_path).sum()))
                            if split == "validation":
                                self.assertAlmostEqual(combined[f"{split}_loss"], restored["loss"])
                                self.assertAlmostEqual(combined[f"{split}_consistency_loss"], restored["consistency_loss"])
                                self.assertAlmostEqual(best.val_coherence_rate, combined["hierarchy"]["coherence_rate"])
                        else:
                            self.assertNotIn("methods", combined)
                            self.assertFalse((run / split / "comparison_metrics.json").exists())
                            self.assertFalse((run / split / "joint_path").exists())
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
                    if len(tasks) == 3:
                        from leaf_hierarchy.hierarchy import decode_logits
                        np.testing.assert_allclose(history.train_loss, history.train_classification_loss
                                                   + history.train_weighted_consistency_loss, rtol=1e-6)
                        np.testing.assert_allclose(history.val_loss, history.val_classification_loss
                                                   + history.val_weighted_consistency_loss, rtol=1e-6)
                        for split in ("validation", "test"):
                            comparison_dir = root / f"hierarchy_{consistency_weight}_{split}"
                            command = ["--checkpoint", str(run / "best.pt"), "--output-dir", str(comparison_dir),
                                       "--device", "cpu", "--num-threads", "2"]
                            if split == "test":
                                command += ["--split", "test"]
                            hierarchy_evaluation.main(command)
                            status = json.loads((comparison_dir / "comparison_config.json").read_text())
                            self.assertEqual(status["status"], "complete")
                            self.assertEqual(status["split"], split)
                            details = next(p for p in comparison_dir.iterdir() if p.is_dir())
                            actual = pd.read_csv(details / "predictions.csv")
                            automatic = pd.read_csv(run / split / "comparison_predictions.csv")
                            pd.testing.assert_frame_equal(actual, automatic)
                            self.assertEqual(json.loads((details / "metrics.json").read_text()),
                                             json.loads((run / split / "comparison_metrics.json").read_text()))
                            original = pd.read_csv(run / split / f"{split}_predictions.csv")
                            self.assertEqual(actual["image_path"].tolist(), original["image_path"].tolist())
                            for task in tasks:
                                self.assertEqual(actual[f"independent_{task}"].tolist(), original[f"predicted_{task}"].tolist())
                            self.assertTrue(actual["species_path_valid_path"].all())
                            self.assertTrue(actual["joint_path_valid_path"].all())
                            with np.load(details / "scores.npz", allow_pickle=False) as scores:
                                with np.load(run / split / "scores.npz", allow_pickle=False) as automatic_scores:
                                    self.assertEqual(set(scores.files), set(automatic_scores.files))
                                    for key in scores.files:
                                        np.testing.assert_array_equal(scores[key], automatic_scores[key])
                                decoded, _ = decode_logits({t: scores[f"logits_{t}"] for t in tasks}, checkpoint["taxonomy"])
                                for task in tasks:
                                    names = scores[f"classes_{task}"]
                                    self.assertEqual(names[decoded["joint_path"][task]].tolist(), actual[f"joint_path_{task}"].tolist())
                            with self.assertRaises(FileExistsError):
                                hierarchy_evaluation.main(command)
                        if consistency_weight == 0:
                            # Reproduce a pre-consistency format-2 checkpoint with no new fields.
                            legacy = torch.load(run / "best.pt", weights_only=True)
                            del legacy["config"]["training"]["consistency_weight"]
                            del legacy["config"]["loss"]["consistency"]
                            old_checkpoint = run / "old.pt"
                            torch.save(legacy, old_checkpoint)
                            legacy_dir = root / "legacy_evaluation"
                            evaluation.main(["--checkpoint", str(old_checkpoint), "--output-dir", str(legacy_dir),
                                             "--config", str(runtime_config), "--data-dir", str(root),
                                             "--split-file", str(manifest), "--device", "cpu", "--num-threads", "2"])
                            old_metrics = json.loads((legacy_dir / "test_metrics.json").read_text())
                            self.assertEqual(old_metrics["test_consistency_weight"], 0.)
                            original_metrics = json.loads((run / "test/test_metrics.json").read_text())
                            self.assertEqual(old_metrics, original_metrics)
            # Keep the legacy command's multi-checkpoint comparison and aggregation contract.
            aggregate_dir = root / "hierarchy_multiple"
            multiple_args = ["--checkpoint", *(str(run / "best.pt") for run in multitask_runs),
                             "--output-dir", str(aggregate_dir), "--device", "cpu", "--num-threads", "2"]
            hierarchy_evaluation.main(multiple_args)
            aggregate = pd.read_csv(aggregate_dir / "aggregate.csv")
            self.assertEqual(len(aggregate), 9)
            self.assertTrue((aggregate.num_runs == 2).all())
            summary = pd.read_csv(aggregate_dir / "summary.csv")
            self.assertEqual(len(summary), 18)
            for _, row in aggregate.iterrows():
                accuracies = [json.loads((run / "validation/comparison_metrics.json").read_text())
                              [row.method]["tasks"][row.task]["accuracy"] for run in multitask_runs]
                self.assertAlmostEqual(row.accuracy_mean, np.mean(accuracies))
                self.assertAlmostEqual(row.accuracy_sd, np.std(accuracies, ddof=1))

            def incompatible_checkpoint(path, device):
                model, checkpoint = load_checkpoint(path, device)
                if Path(path).parent == multitask_runs[1]:
                    checkpoint["preprocessing"] = {**checkpoint["preprocessing"], "crop_size": 32}
                return model, checkpoint

            rejected_args = ["--checkpoint", *(str(run / "best.pt") for run in multitask_runs),
                             "--output-dir", str(root / "incompatible_comparison"),
                             "--device", "cpu", "--num-threads", "2"]
            with patch.object(hierarchy_evaluation, "load_checkpoint", side_effect=incompatible_checkpoint):
                with self.assertRaisesRegex(ValueError, "share.*preprocessing"):
                    hierarchy_evaluation.main(rejected_args)


if __name__ == "__main__":
    unittest.main()
