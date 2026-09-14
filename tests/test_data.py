"""Shared data and label invariants, independent of model architecture."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from leaf_hierarchy.data import (build_taxonomy, load_manifest, make_loader,
    split_fingerprint, taxonomy_fingerprint)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "relocated_dataset"
        (self.data_dir / "images").mkdir(parents=True)
        records = []
        for index, (species, genus, family) in enumerate((
                ("Acer campestre L.", "Acer", "Sapindaceae"),
                ("Quercus robur L.", "Quercus", "Fagaceae"))):
            for split in ("train", "validation", "test"):
                name = f"images/{index}_{split}.jpg"
                Image.new("RGB", (10, 12), (index * 80, 120, 50)).save(self.data_dir / name)
                records.append({"image_path": name, "species": species, "genus": genus,
                    "family": family, "class_id": str(index + 10),
                    "observation_id": f"observation_{index}_{split}",
                    "group_id": f"group_{index}_{split}", "split": split})
        self.frame = pd.DataFrame(records)
        self.manifest = self.root / "split_manifest.csv"

    def load(self, frame=None, mapping=None):
        (self.frame if frame is None else frame).to_csv(self.manifest, index=False)
        return load_manifest(self.manifest, self.data_dir, mapping)

    def test_complete_hierarchy_is_required_and_ambiguities_are_rejected(self):
        for column in ("genus", "family"):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "missing columns"):
                self.load(self.frame.drop(columns=column))
        changed = self.frame.copy()
        changed.loc[0, "genus"] = "Different"
        with self.assertRaisesRegex(ValueError, "multiple genus"):
            self.load(changed)
        changed = self.frame.copy()
        changed.loc[3:, "genus"] = "Acer"
        with self.assertRaisesRegex(ValueError, "multiple family"):
            self.load(changed)

    def test_observation_group_and_hash_leakage_are_rejected(self):
        for column in ("observation_id", "group_id", "sha256", "pixel_sha256"):
            changed = self.frame.copy()
            if column not in changed:
                changed[column] = [str(i) for i in range(len(changed))]
            changed.loc[1, column] = changed.loc[0, column]
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Data leakage"):
                self.load(changed)

    def test_duplicate_paths_missing_species_in_split_and_changed_class_order_are_rejected(self):
        changed = self.frame.copy()
        changed.loc[1, "image_path"] = changed.loc[0, "image_path"]
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.load(changed)
        with self.assertRaisesRegex(ValueError, "every active species"):
            self.load(self.frame.iloc[1:])
        with self.assertRaisesRegex(ValueError, "class order"):
            self.load(mapping={"Acer campestre L.": 1, "Quercus robur L.": 0})

    def test_taxonomy_is_deterministic_and_preserves_dataset_names(self):
        first = build_taxonomy(self.frame)
        second = build_taxonomy(self.frame.iloc[::-1])
        self.assertEqual(first, second)
        self.assertEqual(first["species_to_genus"]["Acer campestre L."], "Acer")
        self.assertEqual(first["genus_to_family"]["Acer"], "Sapindaceae")
        changed = self.frame.replace({"Sapindaceae": "Changed family"})
        self.assertNotEqual(taxonomy_fingerprint(first), taxonomy_fingerprint(build_taxonomy(changed)))

    def test_loader_returns_each_requested_taxonomic_label_and_keeps_image_order(self):
        data, _ = self.load()
        taxonomy = build_taxonomy(data)
        loader = make_loader(data, "test", self.data_dir, 2, tasks=("species", "genus", "family"),
                             class_mappings=taxonomy["class_mappings"],
                             preprocessing={"resize_size": 12, "crop_size": 8})
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
        self.assertEqual(set(labels), {"species", "genus", "family"})
        for task in labels:
            expected = loader.dataset.frame[task].map(taxonomy["class_mappings"][task]).tolist()
            self.assertEqual(labels[task].tolist(), expected)

    def test_split_identity_ignores_row_order_and_detects_record_changes(self):
        expected = split_fingerprint(self.frame)
        self.assertEqual(split_fingerprint(self.frame.iloc[::-1]), expected)
        for column, value in (("image_path", "images/changed.jpg"), ("species", "Changed species"),
                              ("genus", "Changed genus"), ("family", "Changed family"),
                              ("class_id", "99"), ("observation_id", "changed_observation"),
                              ("group_id", "changed_group"), ("split", "test")):
            changed = self.frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column):
                self.assertNotEqual(split_fingerprint(changed), expected)


if __name__ == "__main__":
    unittest.main()
