"""Check constrained inference against explicit valid-path scores and known errors."""

from copy import deepcopy
import unittest

import numpy as np

from leaf_hierarchy.hierarchy import coherence_flags, compare_predictions, decode_logits, taxonomy_paths
from leaf_hierarchy.hierarchy_evaluation import parse_args


TAXONOMY = {
    "class_mappings": {"family": {"F1": 1, "F0": 0},
                       "genus": {"G2": 2, "G0": 0, "G1": 1},
                       "species": {"S3": 3, "S1": 1, "S0": 0, "S2": 2}},
    "species_to_genus": {"S0": "G0", "S1": "G0", "S2": "G1", "S3": "G2"},
    "genus_to_family": {"G0": "F0", "G1": "F1", "G2": "F1"},
}


class HierarchyTests(unittest.TestCase):
    def test_joint_uses_all_heads_and_saved_class_order(self):
        probabilities = {"family": [[.1, .9]], "genus": [[.1, .8, .1]],
                         "species": [[.45, .05, .4, .1]]}
        predictions, scores = decode_logits({k: np.log(v) for k, v in probabilities.items()}, TAXONOMY)
        for method, expected in (("independent", (1, 1, 0)), ("species_path", (0, 0, 0)),
                                  ("joint_path", (1, 1, 2))):
            self.assertEqual(tuple(int(predictions[method][t][0]) for t in ("family", "genus", "species")), expected)
        self.assertFalse(coherence_flags(predictions["independent"], TAXONOMY)["valid_path"][0])
        for method in ("species_path", "joint_path"):
            self.assertTrue(coherence_flags(predictions[method], TAXONOMY)["valid_path"].all())
        explicit_products = [.1*.1*.45, .1*.1*.05, .9*.8*.4, .9*.1*.1]
        self.assertEqual(predictions["joint_path"]["species"][0], np.argmax(explicit_products))

    def test_coherent_argmax_is_preserved_and_ties_are_deterministic(self):
        logits = {"family": [[4, 0], [0, 0]], "genus": [[4, 0, 0], [0, 0, 0]],
                  "species": [[0, 4, 0, 0], [0, 0, 0, 0]]}
        predictions, _ = decode_logits(logits, TAXONOMY)
        for method in predictions:
            np.testing.assert_array_equal(predictions[method]["species"], [1, 0])

    def test_joint_scores_are_stable_for_extreme_logits(self):
        logits = {"family": [[1e5, -1e5]], "genus": [[1e5, -1e5, 0]],
                  "species": [[-1e5, 1e5, 0, 0]]}
        predictions, scores = decode_logits(logits, TAXONOMY)
        self.assertEqual(predictions["joint_path"]["species"][0], 1)
        self.assertTrue(all(np.isfinite(v).all() for v in scores.values()))

    def test_valid_paths_can_be_wrong_and_changes_can_help_or_hurt(self):
        logits = {"family": np.log([[.1, .9]]*2), "genus": np.log([[.1, .8, .1]]*2),
                  "species": np.log([[.45, .05, .4, .1]]*2)}
        predictions, _ = decode_logits(logits, TAXONOMY)
        truth = {"family": np.array([1, 0]), "genus": np.array([1, 0]), "species": np.array([2, 0])}
        metrics = compare_predictions(truth, predictions, TAXONOMY)
        self.assertEqual(metrics["independent"]["invalid_paths"], 2)
        for method in ("species_path", "joint_path"):
            self.assertEqual(metrics[method]["coherence_rate"], 1.)
            self.assertEqual(metrics[method]["complete_path_accuracy"], .5)
            self.assertEqual(metrics[method]["tasks"]["species"]["accuracy"], .5)
        joint = metrics["joint_path"]["tasks"]["species"]
        self.assertEqual(joint["fixed_from_independent"], 1)
        self.assertEqual(joint["harmed_from_independent"], 1)
        self.assertAlmostEqual(joint["macro_f1"], 1/6)  # All four classes, including absent ones.
        self.assertEqual(metrics["species_path"]["tasks"]["species"]["changed_from_independent"], 0)

    def test_invalid_taxonomy_and_scores_fail(self):
        broken = deepcopy(TAXONOMY)
        broken["species_to_genus"]["S0"] = "unknown"
        with self.assertRaisesRegex(ValueError, "relationship"):
            taxonomy_paths(broken)
        valid = {"family": np.zeros((1, 2)), "genus": np.zeros((1, 3)), "species": np.zeros((1, 4))}
        for bad in ({"species": valid["species"]}, {**valid, "family": [[float('nan'), 0]]},
                    {**valid, "species": np.zeros((2, 4))}, {**valid, "genus": np.zeros((1, 5))},
                    {t: v[:0] for t, v in valid.items()}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                decode_logits(bad, TAXONOMY)

    def test_validation_is_default_and_training_is_not_supported(self):
        self.assertEqual(parse_args(["--checkpoint", "best.pt"] ).split, "validation")
        self.assertEqual(parse_args(["--checkpoint", "first.pt", "second.pt", "--split", "test"]).split, "test")


if __name__ == "__main__":
    unittest.main()
