# ResNet18 family, genus and species multitask experiment

This experiment tests whether joint supervision at three taxonomic levels improves
species classification over the [species baseline](../resnet18_species/README.md).
One ResNet18 produces a shared 512-dimensional embedding. Three independent linear
heads predict family, genus and species from that embedding. All parameters train
together with AdamW.

## Protocol

[config.toml](config.toml) keeps the baseline's manifest, ImageNet initialization,
five epochs, batch size 16, seed 20260910, learning rate 0.0001, weight decay 0.0001,
preprocessing and augmentation. Class counts and indices come from the manifest's
taxonomy. The current prepared dataset has 21 families, 32 genera and 41 species.
Dataset taxonomic names are preserved, including the discrepancy documented in the
[dataset guide](../../PlantCLEF2015TrainingData/README.md#taxonomic-labels).

The objective is `CE(family) + CE(genus) + CE(species)`, with each cross-entropy
averaged over the batch and each task weighted 1. The run configuration records
this loss definition. The total multitask loss has a different scale from the
baseline loss; compare the separately reported species loss, accuracy and macro F1.
Each head predicts independently; taxonomic consistency is not enforced.

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
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_multitask/RUN_ID/best.pt
leaf-hierarchy predict --checkpoint results/runs/resnet18_multitask/RUN_ID/best.pt --image IMAGE --json
```

Prediction restores all three heads and their vocabularies from the checkpoint.
It returns a label and softmax score for each level.

## Saved metrics

`history.csv` records total training loss, total validation loss and independent
`val_family_*`, `val_genus_*` and `val_species_*` columns for loss, accuracy and
macro F1 at every epoch. `config.json` records settings, input fingerprints,
class mappings, software versions and source identity. `best.pt` stores the
selected weights and validation metrics.

Training writes `validation/` whenever it selects a better checkpoint. Evaluation
writes `test/` for the checkpoint supplied. Both directories use the following
layout, with `SPLIT` equal to `validation` or `test`:

| Artifact | Contents |
| --- | --- |
| `SPLIT_family_metrics.json` | Family loss, accuracy, macro F1, class count, image count and selected epoch |
| `SPLIT_genus_metrics.json` | The same measurements for genus |
| `SPLIT_species_metrics.json` | The same measurements for species |
| `SPLIT_metrics.json` | Combined results grouped by task and total loss |
| `SPLIT_TASK_classification_report.csv` | Per-class precision, recall, F1 and support for each task |
| `SPLIT_TASK_confusion_matrix.csv` | One confusion matrix per task, true labels in rows |
| `SPLIT_predictions.csv` | Image identities, true labels and three predicted labels |

Test evaluation also writes `evaluation_config.json` with the checkpoint path and
hash, input paths and execution settings. Validation artifacts correspond to
`best.pt`, which can come from an earlier epoch than the last row in `history.csv`.
Accuracy and macro F1 are image-level measures. Macro F1 includes every class in
the saved vocabulary and uses zero for undefined per-class scores.

## Comparison with the baseline

Use exactly the same manifest and taxonomy for both experiments. Verify the saved
`split_records_sha256`, class mappings, preprocessing, training settings and
software environment. The species baseline remains available with its original
configuration and checkpoint format. New baseline runs also save separate species
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
metrics, configuration parity with the baseline, and training, checkpoint reload,
validation/test exports and image prediction for both experiments. They use small
synthetic images and random initialization without downloading weights.