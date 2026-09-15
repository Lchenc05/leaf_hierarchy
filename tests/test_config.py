"""Dataset locations remain portable across preparation, training and evaluation."""

import argparse
from contextlib import chdir, ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from leaf_hierarchy import evaluation, training
from leaf_hierarchy.config import apply_overrides, load_config
from leaf_hierarchy.data import plantclef2015
from leaf_hierarchy.runtime import DEFAULT_DATA_DIR, DEFAULT_SPLIT_FILE, ROOT


DATA_DIR_ENV = "LEAF_HIERARCHY_DATA_DIR"


class DatasetLocationTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(DATA_DIR_ENV, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work_dir = Path(temporary.name).resolve()
        self.config_file = self.work_dir / "experiment.toml"
        self.config_file.write_text(
            '[data]\ndata_dir = "datasets/from-config"\n'
            'split_file = "results/custom/split.csv"\n', encoding="utf-8")

    def test_defaults_keep_repository_dataset_and_split(self):
        config = load_config()
        self.assertEqual(Path(config["data"]["data_dir"]), DEFAULT_DATA_DIR)
        self.assertEqual(Path(config["data"]["split_file"]), DEFAULT_SPLIT_FILE)

    def test_toml_relative_paths_use_repository_root_independent_of_cwd(self):
        with chdir(self.work_dir):
            config = load_config(self.config_file)
        self.assertEqual(Path(config["data"]["data_dir"]), ROOT / "datasets/from-config")
        self.assertEqual(Path(config["data"]["split_file"]), ROOT / "results/custom/split.csv")

    def test_toml_accepts_an_absolute_external_dataset_path(self):
        external = self.work_dir / "external dataset"
        self.config_file.write_text(f'[data]\ndata_dir = "{external.as_posix()}"\n',
                                    encoding="utf-8")
        self.assertEqual(Path(load_config(self.config_file)["data"]["data_dir"]), external)

    def test_environment_overrides_toml_and_resolves_relative_to_cwd(self):
        os.environ[DATA_DIR_ENV] = "datasets/from-environment"
        with chdir(self.work_dir):
            config = load_config(self.config_file)
        self.assertEqual(Path(config["data"]["data_dir"]),
                         self.work_dir / "datasets/from-environment")
        self.assertEqual(Path(config["data"]["split_file"]), ROOT / "results/custom/split.csv")

    def test_environment_expands_user_home(self):
        os.environ[DATA_DIR_ENV] = "~/leaf-hierarchy-test-data"
        self.assertEqual(Path(load_config()["data"]["data_dir"]),
                         Path("~/leaf-hierarchy-test-data").expanduser().resolve())

    def test_blank_environment_preserves_configured_or_default_dataset(self):
        for value in ("", " \t "):
            with self.subTest(value=value):
                os.environ[DATA_DIR_ENV] = value
                self.assertEqual(Path(load_config(self.config_file)["data"]["data_dir"]),
                                 ROOT / "datasets/from-config")
                self.assertEqual(Path(load_config()["data"]["data_dir"]), DEFAULT_DATA_DIR)

    def test_cli_overrides_environment_and_toml_without_mutating_loaded_config(self):
        os.environ[DATA_DIR_ENV] = str(self.work_dir / "environment")
        original = load_config(self.config_file)
        with chdir(self.work_dir):
            changed = apply_overrides(original, argparse.Namespace(data_dir=Path("cli data")))
        self.assertEqual(Path(changed["data"]["data_dir"]), self.work_dir / "cli data")
        self.assertEqual(Path(original["data"]["data_dir"]), self.work_dir / "environment")

    def test_prepare_uses_environment_without_an_experiment_config(self):
        os.environ[DATA_DIR_ENV] = str(self.work_dir / "external dataset")
        with patch.object(plantclef2015, "run_analysis") as analysis:
            self.assertEqual(plantclef2015.main([]), 0)
        analysis.assert_called_once_with(
            self.work_dir / "external dataset", ROOT / "results/data/plantclef2015/v1",
            ROOT / "PlantCLEF2015TrainingData/taxonomy_snapshot.json", 8,
            manifest_name="split_manifest.csv",
        )

    def test_prepare_passes_resolved_location_and_keeps_repository_taxonomy(self):
        cases = (
            (None, [], ROOT / "datasets/from-config"),
            ("environment data", [], self.work_dir / "environment data"),
            ("environment data", ["--data-dir", "cli data"], self.work_dir / "cli data"),
        )
        output = self.work_dir / "prepared"
        taxonomy = ROOT / "PlantCLEF2015TrainingData/taxonomy_snapshot.json"
        for environment, extra_args, expected in cases:
            with self.subTest(environment=environment, extra_args=extra_args):
                if environment is None:
                    os.environ.pop(DATA_DIR_ENV, None)
                else:
                    os.environ[DATA_DIR_ENV] = environment
                with chdir(self.work_dir), patch.object(plantclef2015, "run_analysis") as analysis:
                    result = plantclef2015.main([
                        "--config", str(self.config_file), "--output-dir", str(output),
                        "--workers", "2", *extra_args,
                    ])
                self.assertEqual(result, 0)
                analysis.assert_called_once_with(expected, output, taxonomy, 2,
                                                 manifest_name="split.csv")

    def test_prepare_uses_configured_split_and_ignores_other_experiment_settings(self):
        self.config_file.write_text(
            self.config_file.read_text(encoding="utf-8")
            + '\n[model]\narchitecture = "future-model"\n'
              '[training]\nepochs = -1\nunsupported_setting = true\n'
              '[runtime]\nnum_threads = 0\n', encoding="utf-8")
        with patch.object(plantclef2015, "run_analysis") as analysis:
            self.assertEqual(plantclef2015.main(["--config", str(self.config_file)]), 0)
        analysis.assert_called_once_with(
            ROOT / "datasets/from-config", ROOT / "results/custom",
            ROOT / "PlantCLEF2015TrainingData/taxonomy_snapshot.json", 8,
            manifest_name="split.csv",
        )

    def test_prepare_rejects_invalid_data_settings_before_analysis(self):
        cases = (
            'data = "dataset"\n',
            '[data]\ndata_dir = 16\n',
            '[data]\nsplit_file = " "\n',
            '[data]\ndata_directory = "dataset"\n',
        )
        for contents in cases:
            with self.subTest(contents=contents):
                self.config_file.write_text(contents, encoding="utf-8")
                errors = StringIO()
                with redirect_stderr(errors), patch.object(plantclef2015, "run_analysis") as analysis:
                    with self.assertRaises(SystemExit) as raised:
                        plantclef2015.main(["--config", str(self.config_file)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("data", errors.getvalue())
                analysis.assert_not_called()

    def test_training_keeps_optimizer_settings_in_toml_with_quick_cli_overrides(self):
        self.config_file.write_text(
            '[training]\nlr = 0.003\nweight_decay = 0.02\nepochs = 12\n',
            encoding="utf-8")
        args = training.parse_args(["--config", str(self.config_file), "--epochs", "1"])
        config = apply_overrides(load_config(args.config), args)
        self.assertEqual(config["training"]["lr"], 0.003)
        self.assertEqual(config["training"]["weight_decay"], 0.02)
        self.assertEqual(config["training"]["epochs"], 1)

    def evaluate_paths(self, *arguments):
        """Stop at the first dataset read and expose its paths to each scenario."""
        checkpoint = {
            "config": {"data": {"data_dir": str(self.work_dir / "saved-data"),
                                "split_file": str(self.work_dir / "saved-split.csv")}},
            "model_spec": {"tasks": ["species"]},
            "taxonomy": {"class_mappings": {"species": {"Acer campestre L.": 0}}},
        }
        stop_before_dataset_read = RuntimeError("dataset path captured")
        with ExitStack() as stack:
            stack.enter_context(chdir(self.work_dir))
            stack.enter_context(patch.object(evaluation, "select_device", return_value="cpu"))
            stack.enter_context(patch.object(evaluation, "configure_runtime"))
            stack.enter_context(patch.object(evaluation, "load_checkpoint", return_value=(None, checkpoint)))
            manifest = stack.enter_context(patch.object(
                evaluation, "load_manifest", side_effect=stop_before_dataset_read))
            with self.assertRaises(RuntimeError) as raised:
                evaluation.main(["--checkpoint", "saved-checkpoint.pt", *arguments])
            self.assertIs(raised.exception, stop_before_dataset_read)
        manifest.assert_called_once()
        split_file, data_dir, mapping = manifest.call_args.args
        self.assertEqual(mapping, checkpoint["taxonomy"]["class_mappings"]["species"])
        return data_dir, split_file

    def test_evaluation_uses_checkpoint_locations_when_no_override_is_set(self):
        for environment in (None, "", " \t "):
            with self.subTest(environment=environment):
                if environment is None:
                    os.environ.pop(DATA_DIR_ENV, None)
                else:
                    os.environ[DATA_DIR_ENV] = environment
                self.assertEqual(self.evaluate_paths(),
                                 (self.work_dir / "saved-data", self.work_dir / "saved-split.csv"))

    def test_evaluation_environment_relocates_images_and_preserves_saved_split(self):
        os.environ[DATA_DIR_ENV] = "relocated dataset"
        self.assertEqual(self.evaluate_paths(),
                         (self.work_dir / "relocated dataset", self.work_dir / "saved-split.csv"))

    def test_evaluation_cli_relocates_images_and_can_override_saved_split(self):
        os.environ[DATA_DIR_ENV] = "environment data"
        self.assertEqual(self.evaluate_paths("--data-dir", "cli data"),
                         (self.work_dir / "cli data", self.work_dir / "saved-split.csv"))
        self.assertEqual(self.evaluate_paths("--data-dir", "cli data", "--split-file", "cli.csv"),
                         (self.work_dir / "cli data", self.work_dir / "cli.csv"))

    def test_evaluation_explicit_config_controls_both_paths(self):
        self.assertEqual(self.evaluate_paths("--config", str(self.config_file)),
                         (ROOT / "datasets/from-config", ROOT / "results/custom/split.csv"))

    def test_evaluation_environment_overrides_explicit_config_dataset_only(self):
        os.environ[DATA_DIR_ENV] = "relocated dataset"
        self.assertEqual(self.evaluate_paths("--config", str(self.config_file)),
                         (self.work_dir / "relocated dataset", ROOT / "results/custom/split.csv"))


class PreparationOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work_dir = Path(temporary.name).resolve()
        self.data_dir = self.work_dir / "dataset"
        self.data_dir.mkdir()
        (self.data_dir / "sample.xml").write_text("<sample/>", encoding="utf-8")
        self.taxonomy_file = self.work_dir / "snapshot.json"
        self.taxonomy_file.write_text("{}", encoding="utf-8")
        self.output_dir = self.work_dir / "prepared"
        self.proposal = pd.DataFrame([{
            "image_path": "PlantCLEF2015TrainingData/train/leaf.jpg",
            "species": "Acer campestre L.", "genus": "Acer", "family": "Sapindaceae",
            "class_id": "10", "observation_id": "observation_1", "group_id": "group_1",
            "split": "train", "content": "LeafScan",
        }])

    def prepare_synthetic_split(self, manifest_name):
        """Exercise output publication with the costly dataset analysis replaced."""
        def write_proposal(inventory, selected, summary, exclusions, data_dir, output_dir, config, workers):
            self.proposal.to_csv(output_dir / "split_manifest.csv", index=False)
            return self.proposal

        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(patch.object(plantclef2015, "inventory_dataset", return_value=self.proposal))
            stack.enter_context(patch.object(
                plantclef2015, "select_tree_leafscans", return_value=(self.proposal, pd.DataFrame())))
            stack.enter_context(patch.object(plantclef2015, "summarize_dataset", return_value=self.proposal))
            stack.enter_context(patch.object(plantclef2015, "build_split_proposal", side_effect=write_proposal))
            return plantclef2015.run_analysis(
                self.data_dir, self.output_dir, self.taxonomy_file, 1, manifest_name=manifest_name)

    def test_custom_manifest_is_written_with_matching_metadata(self):
        self.prepare_synthetic_split("custom_split.csv")
        saved = pd.read_csv(self.output_dir / "custom_split.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(saved, self.proposal)
        self.assertFalse((self.output_dir / "split_manifest.csv").exists())
        config = json.loads((self.output_dir / "config.json").read_text(encoding="utf-8"))
        schema = json.loads((self.output_dir / "manifest_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(config["manifest_filename"], "custom_split.csv")
        self.assertEqual(schema["active_split_records_sha256"], plantclef2015.split_fingerprint(saved))
        self.assertTrue((self.output_dir / "taxonomy.json").is_file())

    def test_conflicting_manifest_name_fails_without_publishing_incomplete_outputs(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.prepare_synthetic_split("config.json")
        self.assertFalse(self.output_dir.exists())
        self.assertEqual(list(self.work_dir.glob(".prepared-*")), [])


if __name__ == "__main__":
    unittest.main()
