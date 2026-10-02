"""Check hierarchical consistency against known marginals and loss gradients."""

from copy import deepcopy
import math
import unittest

import torch

from leaf_hierarchy.consistency import (
    HierarchicalConsistencyLoss, loss_metadata, saved_consistency_weight, validate_consistency_weight,
)
from leaf_hierarchy.engine import evaluate_loader, loss_components, task_loss
from leaf_hierarchy.hierarchy import coherence_flags


TASKS = ("family", "genus", "species")
# Neither insertion order nor alphabetical order is the saved class-index order.
TAXONOMY = {
    "class_mappings": {
        "family": {"A_family": 1, "Z_family": 0},
        "genus": {"A_genus": 2, "Z_genus": 0, "M_genus": 1},
        "species": {"A_species": 3, "Z_species": 0, "M_species": 1, "B_species": 2},
    },
    "species_to_genus": {
        "A_species": "A_genus", "Z_species": "Z_genus",
        "M_species": "Z_genus", "B_species": "M_genus",
    },
    "genus_to_family": {"A_genus": "A_family", "Z_genus": "Z_family", "M_genus": "A_family"},
}


def logits_from_probabilities(probabilities, dtype=torch.float64):
    return {task: torch.tensor(values, dtype=dtype).log().requires_grad_()
            for task, values in probabilities.items()}


def inconsistent_logits(dtype=torch.float64):
    return logits_from_probabilities({
        "family": [[.8, .2], [.1, .9]],
        "genus": [[.1, .7, .2], [.6, .1, .3]],
        "species": [[.4, .1, .3, .2], [.1, .2, .2, .5]],
    }, dtype)


def js_from_entropy(first, second):
    """Independent numeric oracle: JS = H((p+q)/2)-(H(p)+H(q))/2."""
    def entropy(values):
        return -sum(value * math.log(value) for value in values if value)

    middle = [(p + q) / 2 for p, q in zip(first, second)]
    return entropy(middle) - (entropy(first) + entropy(second)) / 2


class ConsistencyTests(unittest.TestCase):
    def test_matches_explicit_entropy_formula_and_saved_class_indices(self):
        # Species grouped into genera: [.4+.1, .3, .2] and [.1+.2, .2, .5].
        # Genus grouped into families: [.1, .7+.2] and [.6, .1+.3].
        expected = (
            js_from_entropy([.1, .7, .2], [.5, .3, .2]) + js_from_entropy([.8, .2], [.1, .9])
            + js_from_entropy([.6, .1, .3], [.3, .2, .5]) + js_from_entropy([.1, .9], [.6, .4])
        ) / 4
        criterion = HierarchicalConsistencyLoss(TAXONOMY)
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                loss = criterion(inconsistent_logits(dtype))
                self.assertEqual(loss.ndim, 0)
                self.assertEqual(loss.dtype, dtype)
                self.assertAlmostEqual(loss.item(), expected, delta=tolerance)

    def test_zero_for_compatible_marginals_does_not_guarantee_coherent_argmax(self):
        outputs = logits_from_probabilities({
            "species": [[.30, .25, .35, .10]],
            "genus": [[.55, .35, .10]],
            "family": [[.55, .45]],
        })
        self.assertAlmostEqual(HierarchicalConsistencyLoss(TAXONOMY)(outputs).item(), 0., delta=1e-12)
        # The most probable species belongs to M_genus, but Z_genus has more total mass.
        predicted = {task: values.detach().argmax(1).numpy() for task, values in outputs.items()}
        self.assertFalse(coherence_flags(predicted, TAXONOMY)["valid_path"][0])

    def test_gradients_reach_every_head_and_match_finite_differences(self):
        criterion = HierarchicalConsistencyLoss(TAXONOMY)
        outputs = inconsistent_logits()
        inputs = tuple(outputs[task] for task in TASKS)
        self.assertTrue(torch.autograd.gradcheck(
            lambda *values: criterion(dict(zip(TASKS, values))), inputs,
            eps=1e-6, atol=1e-5, rtol=1e-4,
        ))
        loss = criterion(outputs)
        gradients = torch.autograd.grad(loss, inputs)
        for task, gradient in zip(TASKS, gradients):
            with self.subTest(task=task):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.norm().item(), 0.)
        stepped = {task: value.detach() - .1 * gradient
                   for task, value, gradient in zip(TASKS, inputs, gradients)}
        self.assertLess(criterion(stepped).item(), loss.item())

    def test_extreme_logits_have_bounded_finite_loss_and_gradients(self):
        criterion = HierarchicalConsistencyLoss(TAXONOMY)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # Both adjacent pairs have disjoint limiting distributions: JS -> ln(2).
                outputs = {
                    "family": torch.tensor([[1e5, -1e5]], dtype=dtype, requires_grad=True),
                    "genus": torch.tensor([[-1e5, 1e5, -1e5]], dtype=dtype, requires_grad=True),
                    "species": torch.tensor([[1e5, -1e5, -1e5, -1e5]], dtype=dtype, requires_grad=True),
                }
                loss = criterion(outputs)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreaterEqual(loss.item(), 0.)
                self.assertLessEqual(loss.item(), math.log(2) + 1e-6)
                self.assertAlmostEqual(loss.item(), math.log(2), delta=1e-6)
                loss.backward()
                for values in outputs.values():
                    self.assertIsNotNone(values.grad)
                    self.assertTrue(torch.isfinite(values.grad).all())

    def test_parents_without_represented_children_have_zero_marginal_mass(self):
        taxonomy = deepcopy(TAXONOMY)
        taxonomy["class_mappings"]["genus"]["empty_genus"] = 3
        taxonomy["genus_to_family"]["empty_genus"] = "Z_family"
        taxonomy["class_mappings"]["family"]["empty_family"] = 2
        criterion = HierarchicalConsistencyLoss(taxonomy)
        expected = .5 * (
            js_from_entropy([.2, .3, .1, .4], [.5, .3, .2, 0.])
            + js_from_entropy([.4, .4, .2], [.6, .4, 0.])
        )
        for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-12)):
            with self.subTest(dtype=dtype):
                outputs = logits_from_probabilities({
                    "family": [[.4, .4, .2]], "genus": [[.2, .3, .1, .4]],
                    "species": [[.4, .1, .3, .2]],
                }, dtype)
                loss = criterion(outputs)
                self.assertTrue(torch.isfinite(loss))
                self.assertAlmostEqual(loss.item(), expected, delta=tolerance)
                loss.backward()
                for values in outputs.values():
                    self.assertIsNotNone(values.grad)
                    self.assertTrue(torch.isfinite(values.grad).all())

    def test_zero_weight_preserves_exact_classification_loss_and_gradients(self):
        targets = {"family": torch.tensor([0, 1]), "genus": torch.tensor([0, 2]),
                   "species": torch.tensor([1, 3])}
        weights = {"family": .5, "genus": 2., "species": 1.}
        for consistency in (None, HierarchicalConsistencyLoss(TAXONOMY)):
            with self.subTest(diagnostics=consistency is not None):
                outputs = inconsistent_logits()
                original = task_loss(outputs, targets, TASKS, weights)
                result = loss_components(outputs, targets, TASKS, weights, consistency=consistency)
                self.assertIs(result["loss"], result["classification_loss"])
                self.assertTrue(torch.equal(result["loss"], original))
                self.assertEqual(result["weighted_consistency_loss"].item(), 0.)
                self.assertEqual(result["consistency_loss"].item() > 0., consistency is not None)
                inputs = tuple(outputs[task] for task in TASKS)
                expected = torch.autograd.grad(original, inputs, retain_graph=True)
                actual = torch.autograd.grad(result["loss"], inputs)
                for old, new in zip(expected, actual):
                    self.assertTrue(torch.equal(old, new))

        species_outputs = {"species": inconsistent_logits()["species"]}
        species_targets = {"species": targets["species"]}
        result = loss_components(species_outputs, species_targets, ("species",))
        self.assertIs(result["loss"], result["classification_loss"])
        self.assertTrue(torch.equal(result["loss"], torch.nn.functional.cross_entropy(
            species_outputs["species"], species_targets["species"])))
        self.assertEqual(result["consistency_loss"].item(), 0.)

    def test_positive_weight_adds_consistency_with_correct_gradient_scale(self):
        outputs = inconsistent_logits()
        targets = {"family": torch.tensor([0, 1]), "genus": torch.tensor([0, 2]),
                   "species": torch.tensor([1, 3])}
        criterion = HierarchicalConsistencyLoss(TAXONOMY)
        result = loss_components(outputs, targets, TASKS, consistency=criterion, consistency_weight=.3)
        classification = task_loss(outputs, targets, TASKS)
        hierarchy = criterion(outputs)
        torch.testing.assert_close(result["classification_loss"], classification)
        torch.testing.assert_close(result["consistency_loss"], hierarchy)
        torch.testing.assert_close(result["weighted_consistency_loss"], .3 * hierarchy)
        torch.testing.assert_close(result["loss"], classification + .3 * hierarchy)
        inputs = tuple(outputs[task] for task in TASKS)
        ce_gradients = torch.autograd.grad(classification, inputs, retain_graph=True)
        hierarchy_gradients = torch.autograd.grad(hierarchy, inputs, retain_graph=True)
        gradients = torch.autograd.grad(result["loss"], inputs)
        for actual, ce_gradient, hierarchy_gradient in zip(gradients, ce_gradients, hierarchy_gradients):
            torch.testing.assert_close(actual, ce_gradient + .3 * hierarchy_gradient)

    def test_invalid_taxonomy_is_rejected(self):
        missing_head = deepcopy(TAXONOMY)
        del missing_head["class_mappings"]["family"]
        duplicate_index = deepcopy(TAXONOMY)
        duplicate_index["class_mappings"]["genus"]["A_genus"] = 0
        missing_edge = deepcopy(TAXONOMY)
        del missing_edge["species_to_genus"]["A_species"]
        unknown_parent = deepcopy(TAXONOMY)
        unknown_parent["genus_to_family"]["A_genus"] = "unknown"
        for taxonomy in (missing_head, duplicate_index, missing_edge, unknown_parent):
            with self.subTest(taxonomy=taxonomy), self.assertRaises(ValueError):
                HierarchicalConsistencyLoss(taxonomy)

    def test_invalid_logits_are_rejected(self):
        criterion = HierarchicalConsistencyLoss(TAXONOMY)
        valid = inconsistent_logits()
        invalid = [
            {"species": valid["species"]},
            {**valid, "extra": valid["family"]},
            {**valid, "family": torch.zeros(2, 3)},
            {**valid, "genus": torch.zeros(3, 3)},
            {**valid, "species": torch.zeros(4)},
            {task: values[:0] for task, values in valid.items()},
            {**valid, "family": torch.tensor([[float("nan"), 0.], [0., 0.]])},
            {**valid, "family": torch.tensor([[float("inf"), 0.], [0., 0.]])},
        ]
        for index, outputs in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                criterion(outputs)

    def test_evaluation_averages_samples_and_reports_independent_coherence(self):
        outputs = logits_from_probabilities({
            "family": [[.8, .2], [.1, .9], [.1, .9]],
            "genus": [[.1, .7, .2], [.6, .1, .3], [.1, .2, .7]],
            "species": [[.4, .1, .3, .2], [.1, .2, .2, .5], [.1, .1, .2, .6]],
        })
        targets = {"family": torch.tensor([0, 1, 1]), "genus": torch.tensor([0, 2, 1]),
                   "species": torch.tensor([1, 3, 2])}

        class IndexedOutputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for task, values in outputs.items():
                    self.register_buffer("logits_" + task, values.detach())

            def forward(self, indices):
                return {task: getattr(self, "logits_" + task)[indices] for task in TASKS}

        dataset = [(torch.tensor(index), {task: values[index] for task, values in targets.items()})
                   for index in range(3)]
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        model = IndexedOutputs()
        result = evaluate_loader(model, loader, torch.device("cpu"), tasks=TASKS,
                                 class_mappings=TAXONOMY["class_mappings"], taxonomy=TAXONOMY,
                                 consistency_weight=.3)
        expected = loss_components(outputs, targets, TASKS,
                                   consistency=HierarchicalConsistencyLoss(TAXONOMY), consistency_weight=.3)
        for name in ("loss", "classification_loss", "consistency_loss", "weighted_consistency_loss"):
            with self.subTest(component=name):
                self.assertAlmostEqual(result[name], expected[name].item(), delta=1e-12)
        self.assertFalse(model.training)
        self.assertEqual(result["consistency_weight"], .3)
        for task in TASKS:
            self.assertEqual(result["true_labels"][task], targets[task].tolist())
            self.assertEqual(result["predicted_labels"][task], outputs[task].argmax(1).tolist())
            self.assertAlmostEqual(result["tasks"][task]["loss"],
                                   torch.nn.functional.cross_entropy(outputs[task], targets[task]).item(),
                                   delta=1e-12)
        hierarchy = result["hierarchy"]
        self.assertEqual(hierarchy["images"], 3)
        self.assertEqual(hierarchy["valid_paths"], 1)
        self.assertEqual(hierarchy["invalid_paths"], 2)
        self.assertAlmostEqual(hierarchy["coherence_rate"], 1 / 3)
        for name in ("valid_path", "species_genus_consistent", "genus_family_consistent"):
            self.assertEqual(result["coherence_flags"][name].tolist(), [False, False, True])
        self.assertEqual(result["coherence_flags"]["species_family_consistent"].tolist(), [True, True, True])
        self.assertEqual(hierarchy["pair_consistency"], {
            "species_genus_consistent": 1 / 3, "genus_family_consistent": 1 / 3,
            "species_family_consistent": 1.,
        })
        # Equal vocabulary sizes do not make a different saved class ordering compatible.
        mismatched = deepcopy(TAXONOMY["class_mappings"])
        mismatched["family"] = {"A_family": 0, "Z_family": 1}
        with self.assertRaisesRegex(ValueError, "(?i)(mapping|disagree)"):
            evaluate_loader(model, loader, torch.device("cpu"), tasks=TASKS,
                            class_mappings=mismatched, taxonomy=TAXONOMY, consistency_weight=.3)

    def test_saved_objective_restores_weight_and_rejects_conflicting_metadata(self):
        self.assertEqual(saved_consistency_weight({}, TASKS), 0.)
        self.assertEqual(saved_consistency_weight({"loss": {"name": "cross_entropy"}}, TASKS), 0.)
        self.assertEqual(saved_consistency_weight({}, ("species",)), 0.)
        config = {"training": {"consistency_weight": .3}, "loss": loss_metadata(TASKS, .3)}
        self.assertEqual(saved_consistency_weight(config, TASKS), .3)
        wrong_version = deepcopy(config)
        wrong_version["loss"]["consistency"]["version"] = 2
        wrong_weight = deepcopy(config)
        wrong_weight["loss"]["consistency"]["weight"] = .7
        missing_metadata = deepcopy(config)
        del missing_metadata["loss"]["consistency"]
        for broken in (wrong_version, wrong_weight, missing_metadata):
            with self.subTest(config=broken), self.assertRaisesRegex(ValueError, "(?i)(metadata|objective)"):
                saved_consistency_weight(broken, TASKS)

    def test_weight_requires_finite_nonnegative_number_and_full_multitask(self):
        for value in (0, 0., .3, 2):
            with self.subTest(value=value):
                actual = validate_consistency_weight(value, TASKS)
                self.assertIsInstance(actual, float)
                self.assertEqual(actual, float(value))
        self.assertEqual(validate_consistency_weight(0., ("species",)), 0.)
        for value in (True, False, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_consistency_weight(value, TASKS)
        with self.assertRaisesRegex(ValueError, "(?i)(family|multitask|tasks|hierarch)"):
            validate_consistency_weight(.1, ("species",))
        with self.assertRaises(ValueError):
            loss_components(inconsistent_logits(), {
                "family": torch.tensor([0, 1]), "genus": torch.tensor([0, 2]),
                "species": torch.tensor([1, 3]),
            }, TASKS, consistency_weight=.3)


if __name__ == "__main__":
    unittest.main()
