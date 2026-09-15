# ResNet18 species baseline

This experiment establishes a species-classification reference for the TFG's later
family/genus/species models. It uses ResNet18 with one species head, initialized from
ImageNet. Hierarchical objectives, representation analyses and Grad-CAM are subsequent
work and are not results of this experiment.

## Environment

Install the package using the [project instructions](../../README.md). The historical
reference ran on Windows with 64-bit Python 3.13.9 and CPU-only PyTorch. Its direct
dependency pins are kept in [requirements.txt](requirements.txt); the project's
minimum Python version is separate from this recorded environment.

To install those dependencies on Windows or Linux with CPU-only PyTorch, run from
the repository root in an activated environment:

```text
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/resnet18_species/requirements.txt
python -m pip install -e .
python -m pip check
```

On macOS, omit the command containing `--index-url`. For an NVIDIA GPU, use the
matching PyTorch 2.10.0 / torchvision 0.25.0 CUDA installation command from the
[official instructions](https://pytorch.org/get-started/previous-versions/#v2100)
before installing the remaining dependencies. A compatible NVIDIA driver is required.
`--device auto` selects CUDA when available, otherwise CPU; it does not select MPS.
Changing hardware or platform libraries may affect exact results.

## Data and protocol

Prepare the data using the [dataset guide](../../PlantCLEF2015TrainingData/README.md).
The input manifest is `results/data/plantclef2015/v1/split_manifest.csv`. The original
botanical selection and observation/duplicate groups remain fixed for this reference.

| Setting | Value |
| --- | --- |
| Split seed | `20260909` |
| Allocation | 70% train, 15% validation, 15% test by groups within each species |
| Class eligibility | At least 20 usable groups per species; at least 14/3/3 groups in train/validation/test |
| Model | ResNet18, ImageNet `IMAGENET1K_V1`, one species head |
| Training seed | `20260910` |
| Epochs / batch size | 5 / 16 |
| Optimizer | AdamW, all parameters trained |
| Learning rate / weight decay | `0.0001` / `0.0001` |
| Model selection | Highest validation macro F1, then lower validation loss |

Preprocessing applies EXIF orientation, converts to RGB, resizes to 256, crops to
224 and normalizes with ImageNet statistics. Training adds independent horizontal
and vertical flips, each with probability 0.5. Validation and test preprocessing
have no random augmentation. The configuration records preprocessing and training
settings separately, and checkpoints carry the preprocessing needed for inference.

## Train and evaluate

Run from the repository root:

```text
leaf-hierarchy train --config experiments/resnet18_species/config.toml
```

[config.toml](config.toml) describes the dataset, model, training, runtime, output,
selection, preprocessing and augmentation settings. Paths in that configuration are
relative to the repository root; explicit command-line paths are relative to the
terminal's current directory. Command-line values override the corresponding
configuration values. Unknown configuration keys are rejected.

Set the learning rate and weight decay in `[training]` using `lr` and `weight_decay`.
For a quick trial, override the duration, batch size or device on the command line:

```text
leaf-hierarchy train --config experiments/resnet18_species/config.toml --epochs 1 --batch-size 8 --device cpu
```

ImageNet weights download on first use unless cached. Set `TORCH_HOME` to use an
existing cache; the default cache is `.cache/torch/`. `--weights none` supports an
offline functional check with random initialization. Such a run, or a shorter run,
does not reproduce the five-epoch ImageNet experiment.

Training prints a timestamped directory under `results/runs/resnet18_species/`:

| Artifact | Contents |
| --- | --- |
| `config.json` | Resolved settings, class mapping, input fingerprints and runtime metadata |
| `history.csv` | Training and validation measurements by epoch |
| `best.pt` | Selected model state, architecture, labels, preprocessing and selection metadata |

Automatic training resume is not implemented. An interrupted run can contain partial
artifacts; use a new run for a complete experiment.

Replace `RUN_ID` with the name printed by training:

```text
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_species/RUN_ID/best.pt
```

Evaluation uses the test split and saves these files in the run's `test/` directory:

| Artifact | Contents |
| --- | --- |
| `test_metrics.json` | Selected epoch, test loss, accuracy, macro F1 and support |
| `evaluation_config.json` | Checkpoint path and hash, input paths, runtime settings and preprocessing |
| `test_species_classification_report.csv` | Per-species precision, recall, F1 and support |
| `test_species_confusion_matrix.csv` | True species in rows and predicted species in columns |
| `test_predictions.csv` | Predictions associated with the input records |

Use the same manifest for training and evaluation. For a custom data location and
prepared dataset, edit the existing `[data]` section of `config.toml`:

```toml
[data]
data_dir = "/opt/datasets/PlantCLEF2015TrainingData"
split_file = "results/data/plantclef2015/custom/split_manifest.csv"
```

On Windows, use a path such as `"D:/datasets/PlantCLEF2015TrainingData"`.
Prepare this version once, then train and evaluate:

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml
leaf-hierarchy train --config experiments/resnet18_species/config.toml
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_species/RUN_ID/best.pt
```

Preparation reads only `[data]`, writes the manifest at `split_file` and places its
supporting files in the same directory. Skip it if this version already exists.
Evaluation reuses the checkpoint's recorded paths. To read later configuration
changes, pass `--config experiments/resnet18_species/config.toml` to evaluation.
For one-command overrides and a shared location across experiments, see the
[advanced dataset options](../../PlantCLEF2015TrainingData/README.md#advanced-path-options).

Keep all active manifest records and their images available, including train and
validation, for consistency checks. Checkpoints with a saved partition fingerprint
reject changes to the active records. See [existing artifacts](#existing-artifacts)
for historical checkpoints with fewer recorded checks.

Accuracy and macro F1 are image-level metrics. Macro F1 gives each species equal
weight; grouping the split does not make the measurements group-weighted. Select
configurations with validation data and reserve test evaluation for the comparison
protocol.

## Predict an image

Run from the repository root with the environment activated. Replace `RUN_ID`
with the directory name printed by training, or use the full printed checkpoint
path after `--checkpoint`:

```text
leaf-hierarchy predict --checkpoint results/runs/resnet18_species/RUN_ID/best.pt --image IMAGE
```

Replace `IMAGE` with the actual image path. Prediction needs only the
image, checkpoint and installed package; it does not accept an experiment configuration.
The checkpoint supplies the model and preprocessing. Prediction uses `--device auto`
and six CPU threads by default; override these with `--device` and `--num-threads`.
It returns the
predicted species and a softmax score; add `--json` for structured output. The score
is not a calibrated probability of correctness, and predictions are restricted to
the species learned during training.

## Existing artifacts

The manifest reader accepts historical column names, and the checkpoint loader
supports the original saved models. Historical files are optional local artifacts
and are not included in a fresh clone. Obtain the checkpoint and, for evaluation,
its manifest and dataset before using these commands. From the repository root,
evaluate the preserved model and predict an image without training again:

```text
leaf-hierarchy evaluate --checkpoint results/resnet18/20260911_130918_664606/resnet18.pt --split-file results/leafscan/split_manifest.csv --device cpu
leaf-hierarchy predict --checkpoint results/resnet18/20260911_130918_664606/resnet18.pt --image PlantCLEF2015TrainingData/train/100373.jpg --device cpu
```

Evaluation results are saved in `results/resnet18/20260911_130918_664606/test/`.
Preserved run directories are under `results/resnet18/`; their historical manifest
is `results/leafscan/split_manifest.csv`. Fresh preparation writes its manifest to
`results/data/plantclef2015/v1/split_manifest.csv`. Keep those data versions distinct
and use the manifest associated with the checkpoint being evaluated.

The CPU check on 14 September 2026 reproduced test accuracy `0.8697` and macro F1
`0.8834` on 683 images, with epoch 5 selected. The prediction command returned:

```text
Predicted species: Prunus spinosa L.
Softmax score: 0.8699
```

The species matches the image's XML label. This example checks inference; use the
test metrics to assess overall performance. These outputs belong to the historical
checkpoint and are not expected results for every new training run.

For images stored elsewhere, supply `--data-dir`
to evaluation, and supply the actual image path to prediction.
These commands leave the original manifests and checkpoints
in place. To train a new model using the historical partition, run:

```text
leaf-hierarchy train --config experiments/resnet18_species/config.toml --split-file results/leafscan/split_manifest.csv
```

Checkpoints without a saved partition fingerprint emit a warning: class mappings
and group separation are checked, but the original training partition cannot be
verified automatically. Historical manifests also contain fewer diagnostics than
full preparation now generates, including the absence of file and decoded-pixel hashes.

## Historical reference and limitations

The original analysis selected 9,648 readable tree leaf scans from 120 species.
The reference classifier used 4,695 images from 41 species:

| Split | Images | Groups | Species |
| --- | ---: | ---: | ---: |
| Train | 3,369 | 1,263 | 41 |
| Validation | 643 | 271 | 41 |
| Test | 683 | 271 | 41 |

Historical run `20260911_130918_664606` selected epoch 5 and reported validation
accuracy 0.8616, validation macro F1 0.8546, test accuracy **0.8697** and test macro F1
**0.8834**. These are preserved historical measurements. They are not measurements
of a new run of the reorganized pipeline; report a new run from its saved artifacts.

The classes are imbalanced, and 35 of the 41 species have fewer than ten test groups.
Observation and duplicate grouping reduces leakage but cannot establish that separate
observations come from different physical trees. Duplicate detection can miss
transformed copies or group distinct images. The conclusions are limited to the
selected leaf scans; field photographs, unseen species and multitask learning have
not been evaluated in this reference experiment.

The active labels span 32 genera and 21 families in the original dataset taxonomy.
The [dataset guide](../../PlantCLEF2015TrainingData/README.md#taxonomic-labels)
describes their provenance and a family-name discrepancy to resolve before treating
an updated taxonomy as a new experimental target.

## Shared checks

The current suite contains architecture-independent checks for data identity,
taxonomy, label alignment, loss and metrics. It uses synthetic inputs and requires
no trained model or dataset download.

Run the shared checks from the repository root after installing the project:

```text
python -m unittest discover -s tests -v
python -m pip check
```

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing manifest | Run preparation or supply the existing manifest with `--split-file`. |
| Checkpoint/partition mismatch | Use the manifest used to train that checkpoint. |
| CUDA out of memory | Reduce `--batch-size` or select `--device cpu`; retain the changed settings with the run. |
| ImageNet download fails | Restore access or provide the weights in `TORCH_HOME`. |
| Missing package | Activate the intended environment, install the package and run `python -m pip check`. |
