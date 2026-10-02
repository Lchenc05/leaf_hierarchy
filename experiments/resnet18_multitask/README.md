# ResNet18 family, genus and species multitask experiment

This experiment tests whether joint supervision at three taxonomic levels improves
species classification over the [species baseline](../resnet18_species/README.md).
One ResNet18 produces a shared 512-dimensional embedding. Three independent linear
heads predict family, genus and species from that embedding. All parameters train
together with AdamW.

## Protocol

[config.toml](config.toml) uses the baseline's manifest, ImageNet initialization,
five epochs, batch size 16, learning rate 0.0001, weight decay 0.0001,
preprocessing and augmentation. Its configured seed is 20261001; use matching
`--seed` values for paired comparisons with another experiment. Class counts and
indices come from the manifest's taxonomy. The current prepared dataset has 21
families, 32 genera and 41 species.
Dataset taxonomic names are preserved, including the discrepancy documented in the
[dataset guide](../../PlantCLEF2015TrainingData/README.md#taxonomic-labels).

The objective is `CE(family) + CE(genus) + CE(species)`, with each cross-entropy
averaged over the batch and each task weighted 1. The run configuration records
this loss definition. The total multitask loss has a different scale from the
baseline loss; compare the separately reported species loss, accuracy and macro F1.
The training metrics use independent predictions; the objective does not enforce
taxonomic consistency. Evaluation also compares two taxonomic decision rules.
The default `training.consistency_weight` is `0`. The separate
[consistency experiment](../resnet18_consistency/README.md) uses the same model
with an additional training penalty.

The selected checkpoint maximizes **validation species macro F1**, breaking ties
with lower total validation loss. Exact ties retain the earlier epoch. Family and
genus metrics describe that same checkpoint. Test images participate in neither
training nor model selection.

## Train, evaluate and predict

Follow the [installation instructions](../../README.md#installation). Reuse the
baseline's prepared manifest; prepare it only if it does not exist. Set
`[data].data_dir` to the directory containing `train/` if images are stored elsewhere.
Run from the repository root:

```text
leaf-hierarchy train --config experiments/resnet18_multitask/config.toml
```

Training prints a new directory under `results/runs/resnet18_multitask/`.
Replace `RUN_ID` below with its name:

```text
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_multitask/RUN_ID/best.pt --split validation
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_multitask/RUN_ID/best.pt --split test
leaf-hierarchy predict --checkpoint results/runs/resnet18_multitask/RUN_ID/best.pt --image IMAGE --json
```

`evaluate` automatically compares `independent`, `species_path` and `joint_path`
using the same model scores from one image pass. It writes to the chosen split's
directory beside the checkpoint; `--split` defaults to test. Training still selects
`best.pt` using independent validation species macro F1. Evaluating validation
after training adds the three-method comparison for that saved checkpoint.

Prediction restores all three heads and their vocabularies from the checkpoint.
It returns an independent label and softmax score for each level.

## Saved metrics

`history.csv` records total training loss, total validation loss and independent
`val_family_*`, `val_genus_*` and `val_species_*` columns for loss, accuracy and
macro F1 at every epoch. `config.json` records settings, input fingerprints,
class mappings, software versions and source identity. `best.pt` stores the
selected weights and validation metrics.

The history also separates classification loss from consistency loss and its
weighted contribution, and records `val_coherence_rate` and `val_invalid_paths`.
For this experiment's default weight of zero, the total loss equals classification
loss and the weighted consistency contribution is zero. The unweighted
consistency loss remains available as a diagnostic. See the
[field definitions](../resnet18_consistency/README.md#saved-results).

Training writes `validation/` whenever it selects a better checkpoint. Evaluation
writes to `validation/` or `test/` for the checkpoint supplied. Both retain the
following independent metrics, with `SPLIT` equal to `validation` or `test`:

| Artifact | Contents |
| --- | --- |
| `SPLIT_family_metrics.json` | Family loss, accuracy, macro F1, class count, image count and selected epoch |
| `SPLIT_genus_metrics.json` | The same measurements for genus |
| `SPLIT_species_metrics.json` | The same measurements for species |
| `SPLIT_metrics.json` | Per-task metrics, total loss, loss components and independent prediction coherence; evaluation also adds the three-method `methods` object |
| `SPLIT_TASK_classification_report.csv` | Per-class precision, recall, F1 and support for each task |
| `SPLIT_TASK_confusion_matrix.csv` | One confusion matrix per task, true labels in rows |
| `SPLIT_predictions.csv` | Image identities, true labels, three predicted labels and taxonomic consistency flags |

Evaluation also writes `evaluation_config.json` with the checkpoint path and
hash, split, methods, input paths and execution settings. It adds `summary.csv`,
`comparison_metrics.json`, `comparison_predictions.csv`, `scores.npz`, `taxonomy.json`
and one directory per method containing metrics, classification reports and
confusion matrices. See the
[comparison output layout](../hierarchy_inference/README.md#outputs-and-interpretation).
Repeating evaluation replaces its generated files for that split and keeps
unrelated files. Validation artifacts correspond to
`best.pt`, which can come from an earlier epoch than the last row in `history.csv`.
Accuracy and macro F1 are image-level measures. Macro F1 includes every class in
the saved vocabulary and uses zero for undefined per-class scores.

## Comparison with the baseline

Use exactly the same manifest and taxonomy for both experiments. Verify the saved
`split_records_sha256`, class mappings, preprocessing, training settings and
software environment, including the seed for each paired run. The species baseline
remains available with its original configuration and checkpoint format. New baseline runs also save separate species
metrics and selected-checkpoint validation reports.

Compare `tasks.species.accuracy` and `tasks.species.macro_f1` in the two
`test_metrics.json` files, or use their separate `test_species_metrics.json` files
for new runs. Compare validation at each experiment's selected epoch. Report the
two run identities and the differences in species metrics; report family and
genus separately. A single seed does not establish a reliable improvement.

## Verification and results status

```text
python -m unittest discover -s tests -v
```

Tests cover shared-backbone gradients from every head, distinct task losses and
metrics, shared protocol settings with the baseline, and training, checkpoint reload,
validation/test exports and image prediction for both experiments. They use small
synthetic images and random initialization without downloading weights.
