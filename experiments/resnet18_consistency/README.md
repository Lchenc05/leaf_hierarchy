# ResNet18 with taxonomic consistency regularization

This experiment measures how a consistency penalty affects independent taxonomic
predictions and classification performance. It uses the same ResNet18 backbone,
family/genus/species heads and prepared dataset as the
[multitask experiment](../resnet18_multitask/README.md).

## Objective

Each output head produces a softmax distribution. Summing species probabilities
within each genus gives a second distribution over genera; summing genus
probabilities within each family gives a second distribution over families.
The penalty compares these aggregated distributions with the corresponding
parent head:

```text
q_G(g) = sum p_S(s) for species s whose parent is g
q_F(f) = sum p_G(g) for genera g whose parent is f

JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = (p + q) / 2
L_H = (JS(p_G, q_G) + JS(p_F, q_F)) / 2
L = CE_family + CE_genus + CE_species + consistency_weight * L_H
```

Each divergence sums over classes and averages over images in the batch. Logs
are natural, so the unweighted `L_H` is bounded by `ln(2)`. Both sides receive
gradients. The taxonomy and its saved class indices determine the aggregations;
the loss does not infer relationships from label spelling. Every classification
term remains batch-mean cross-entropy with weight 1.

`training.consistency_weight` must be finite and nonnegative. Its default of `0`
preserves the original cross-entropy objective and gradients. A positive value
requires all three output heads and their taxonomy.

This penalty encourages agreement between distributions; it does not guarantee
coherent independent argmax labels. For example, species probabilities
`(0.4, 0.3, 0.3)` can imply genus probabilities `(0.4, 0.6)` when the first species
belongs to one genus and the other two share another. The distributions agree,
but their most likely species and genus belong to different paths.

## Configuration and commands

[config.toml](config.toml) uses ImageNet initialization, five epochs, batch size 16,
seed 20260924, AdamW with learning rate and weight decay both 0.0001, and the
multitask experiment's manifest, preprocessing and augmentation. It sets
`consistency_weight = 1.0` as an initial experimental value, not an optimized value.
The objective components have different scales, so this does not give consistency
the same numerical contribution as the sum of the classification losses.
For a paired comparison, pass the same `--seed` to both training commands; their
current TOML configurations use different seeds.

Run from the repository root with the environment activated and the prepared
dataset available. Set `[data].data_dir` if the images are stored elsewhere:

```text
leaf-hierarchy train --config experiments/resnet18_consistency/config.toml
```

An explicit flag overrides the TOML weight. For example:

```text
leaf-hierarchy train --config experiments/resnet18_consistency/config.toml --consistency-weight 0.5 --seed 20260910
```

Each invocation creates a directory under `results/runs/resnet18_consistency/`.
Replace `RUN_ID` with the directory printed by training:

```text
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_consistency/RUN_ID/best.pt --split validation
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_consistency/RUN_ID/best.pt --split test
```

`evaluate` restores the consistency weight saved in the checkpoint even if a
different `--config` is supplied for data or runtime settings. Older checkpoints
without a saved consistency weight use `0`. The command automatically compares
`independent`, `species_path` and `joint_path` from the same scores in one image
pass, without retraining. `--split` defaults to test. `predict` keeps independent
head predictions. The existing `compare-hierarchy` command remains available for
[multi-checkpoint comparisons](../hierarchy_inference/README.md#run-the-comparison).

The checkpoint maximizes independent validation species macro F1. Within a run,
ties use the lower total validation loss, including the fixed consistency weight; exact ties
retain the earlier epoch. Test images do not affect training or checkpoint
selection. Compare classification metrics and coherence across different weights,
not total loss alone, whose definition changes with the weight. Keep the seed and
other settings matched when comparing runs.

## Saved results

The run contains `config.json`, `history.csv`, `best.pt` and independent
selected-checkpoint results in `validation/`. Running `evaluate --split validation`
adds the three-method comparison for that checkpoint; `--split test` writes to
`test/`. The configuration and checkpoint record the loss definition and resolved
weight. Neither the consistency penalty nor the decoding rules change when
evaluating a different split.

In addition to total losses and per-task validation metrics, `history.csv` includes:

| Columns | Meaning |
| --- | --- |
| `train_classification_loss`, `val_classification_loss` | Sum of the three cross-entropies |
| `train_consistency_loss`, `val_consistency_loss` | Unweighted `L_H` |
| `train_weighted_consistency_loss`, `val_weighted_consistency_loss` | `consistency_weight * L_H` |
| `val_coherence_rate`, `val_invalid_paths` | Coherence fraction and invalid-path count for independent validation predictions |

Losses are averaged per image, including an incomplete final batch. Every
per-task loss remains that task's cross-entropy.

`validation_metrics.json` and `test_metrics.json` include `classification_loss`,
`consistency_loss`, `weighted_consistency_loss` and `consistency_weight`, each
prefixed with `validation_` or `test_`. Their `hierarchy` object contains `images`,
`valid_paths`, `invalid_paths`, `coherence_rate` and `pair_consistency` for the
independent predictions. The prediction CSV adds `valid_path` and compatibility
flags for species-genus, genus-family and species-family. The existing per-task
classification reports and confusion matrices retain their format.

Explicit evaluation adds a `methods` object to the overall metrics JSON, alongside
the original independent metrics and losses. It also saves `summary.csv`,
`comparison_metrics.json`, `comparison_predictions.csv`, `scores.npz`, `taxonomy.json`
and a metrics/report directory for each method. `evaluation_config.json` records
the checkpoint, split and methods. See the
[comparison output layout](../hierarchy_inference/README.md#outputs-and-interpretation)
for details. Repeating evaluation replaces its generated files for the chosen split
and keeps unrelated files.

The unweighted consistency loss is also recorded for multitask runs with weight
zero, allowing comparison against the original objective. Coherence and
classification accuracy measure different properties: a valid taxonomic path can
still be the wrong answer for an image.

## Measured comparison (2 October 2026)

Weights `0` and `1` were compared using paired seeds `20260910`, `20260924` and
`20260927`. All six runs completed five epochs with the same data partition,
taxonomy, preprocessing, augmentation and training settings within each pair,
apart from the consistency weight. Each checkpoint was selected using validation
species macro F1. Weight `1` was the initial setting, not an optimized value.

Test results below are percentages, reported as mean ± sample standard deviation
(`ddof=1`) across the three seeds on the same 683 images:

| Method | Metric | Weight 0 | Weight 1 |
| --- | --- | --- | --- |
| Independent | Species accuracy | 88.58 ± 2.16 | 88.97 ± 0.89 |
| Independent | Species macro F1 | 88.81 ± 2.51 | 89.49 ± 0.70 |
| Independent | Taxonomic coherence | 96.29 ± 0.68 | 96.39 ± 0.30 |
| Joint path | Species accuracy | 88.97 ± 2.64 | 89.56 ± 1.00 |
| Joint path | Species macro F1 | 89.26 ± 3.11 | 90.11 ± 0.53 |

Independent test species accuracy and macro F1 improve for seeds `20260910` and
`20260924`, but decrease for `20260927`. The mean classification improvement is
modest, and independent coherence changes little. Joint paths are 100% coherent
by construction for both weights. Species-derived decoding preserves the
independent species metrics.

On the 643 validation images, independent species macro F1 changes from
85.40 ± 1.32% to 86.02 ± 0.80%; joint-path macro F1 changes from 85.42 ± 0.71%
to 85.61 ± 0.59%. With weight `1`, joint decoding therefore has lower mean
validation macro F1 than independent decoding. Use validation to choose settings
and inference methods. Three seeds do not establish statistical superiority.

The full local tables and source identities are saved in
`results/analysis/consistency/20261002_paired_seeds/report.md` and
`results/analysis/consistency/20261002_paired_seeds/comparison.json`.
These generated files and the checkpoints are excluded from Git.

## Verification

```text
python -m unittest discover -s tests -v
```

The tests use synthetic inputs to check the loss, gradients, taxonomy indexing,
numerical stability, zero-weight compatibility and saved-model workflows. These
checks verify implementation behavior; they do not establish a classification
improvement on the leaf dataset.
