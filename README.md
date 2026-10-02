# leaf_hierarchy

Research code for plant classification from leaf images. The project includes
species-only and multitask ResNet18 models, dataset preparation, training with
optional taxonomic consistency regularization, evaluation and comparison of
inference methods using family, genus and species relationships.

The repository organizes this work as reproducible experiments. Each experiment
documents its research question, data, configuration, evaluation protocol and results.
Shared code handles data validation, model execution and the recording of run artifacts.

## Project organization

The core directories separate reusable code, experiment definitions, generated
artifacts and the thesis document:

```text
leaf_hierarchy/
  src/leaf_hierarchy/       # Shared research code.
    data/                  # Dataset preparation, manifests and taxonomy.
    models/                # Model definitions and access to learned features.
  experiments/             # Experiment catalog, configurations and documentation.
  results/                 # Prepared data and run artifacts; ignored by Git.
  tests/                   # Data, loss, metrics and model workflow checks.
  report/                  # Thesis source and compilation instructions.
  pyproject.toml           # Package metadata, dependencies and command entry point.
```

The [experiment catalog](experiments/README.md) links each experiment's data
preparation guide, configuration, execution instructions and results.
The [thesis guide](report/README.md) covers writing and compiling the report.

## Installation

Use Git and Python 3.11 or later. For the pip installation below, use 64-bit Python:
the published [PyTorch 2.10 packages](https://pypi.org/project/torch/2.10.0/#files)
are available only for 64-bit platforms.

From a terminal:

```text
git clone https://github.com/Lchenc05/leaf_hierarchy.git
cd leaf_hierarchy
```

Create and activate an isolated environment:

```powershell
# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```text
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
```

A GPU is optional. Experiment guides record their reference environments and any
specific dependency or hardware requirements.

## Running experiments

Start with the [ResNet18 species experiment](experiments/resnet18_species/README.md).
Obtain the images using its [dataset guide](PlantCLEF2015TrainingData/README.md), then
set the directory containing `train/` in the existing `[data]` section of
[`experiments/resnet18_species/config.toml`](experiments/resnet18_species/config.toml):

```toml
[data]
data_dir = "/opt/datasets/PlantCLEF2015TrainingData"
split_file = "results/data/plantclef2015/v1/split_manifest.csv"
```

On Windows, use a path such as `"D:/datasets/PlantCLEF2015TrainingData"`.
Keep the default `data_dir` if the dataset is inside the repository. TOML paths are
relative to the repository root unless absolute.

Run from the repository root with the environment activated. Prepare the data once;
skip this command when the configured manifest already exists:

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml
```

Preparation reads only `[data]` and writes the manifest at `split_file`, with its
supporting files in the same directory. Images stay in their original location.

Train using that experiment configuration:

```text
leaf-hierarchy train --config experiments/resnet18_species/config.toml
```

Then evaluate the saved model and predict one image. Replace `CHECKPOINT` with the
path to `best.pt` printed by training and `IMAGE` with your image path:

```text
leaf-hierarchy evaluate --checkpoint CHECKPOINT --split test
leaf-hierarchy predict --checkpoint CHECKPOINT --image IMAGE
```

Evaluation restores the dataset paths saved in the checkpoint. Both evaluation and
prediction restore the model and image preprocessing. Prediction needs no experiment
configuration. The [generated results](#generated-results) section explains the outputs.
The [catalog](experiments/README.md) lists the available experiments.

The [multitask experiment](experiments/resnet18_multitask/README.md) trains family,
genus and species heads together using the same data and baseline settings:

```text
leaf-hierarchy train --config experiments/resnet18_multitask/config.toml
```

Training saves independent validation metrics for each taxonomic level and uses
species macro F1 to select the checkpoint. Run `evaluate` separately to compare
inference methods on that checkpoint in validation or test.

The [consistency experiment](experiments/resnet18_consistency/README.md) adds a
penalty for disagreement between parent probabilities and summed child
probabilities during multitask training:

```text
leaf-hierarchy train --config experiments/resnet18_consistency/config.toml
```

`[training].consistency_weight` controls this penalty. It defaults to `0` for the
original objective; the consistency configuration uses `1` as an initial setting,
not a value selected for optimal performance. Training accepts
`--consistency-weight` to override it. Evaluation restores the weight saved in the
checkpoint, including when `--config` supplies different runtime or data settings.

Use the same evaluation command for every model. Select validation to compare
configurations and decision rules, then test to measure the fixed choices:

```text
leaf-hierarchy evaluate --checkpoint CHECKPOINT --split validation
leaf-hierarchy evaluate --checkpoint CHECKPOINT --split test
```

`--split` defaults to `test`. The checkpoint determines the methods automatically:

| Saved model | Evaluation methods |
| --- | --- |
| Species-only | Independent species predictions |
| Family/genus/species multitask, with or without consistency regularization | `independent`, `species_path` and `joint_path` |

For multitask checkpoints, evaluation compares independent heads, parents derived
from the predicted species and equal-weight joint inference over valid taxonomic
paths. All three methods reuse the same scores from one pass over the images.
Results go beside the checkpoint in `validation/` or `test/`. No method flag is
needed. `predict` continues to use independent output heads.

The existing `compare-hierarchy` command remains available for separate analysis
directories and comparisons across multiple checkpoints. See the
[hierarchical inference experiment](experiments/hierarchy_inference/README.md) for
the fixed protocol, output files and multi-seed comparison.

### Advanced options

Keep experiment settings in the TOML file, including `[training].lr` and
`[training].weight_decay`. For a quick trial, training accepts overrides such as
`--epochs 1`, `--batch-size 8` and `--device cpu`.

Use `--data-dir` to override the dataset location for one command. For a shared
location across experiments and moving an existing checkpoint's dataset, see the
[advanced dataset options](PlantCLEF2015TrainingData/README.md#advanced-path-options).
Use `leaf-hierarchy --help` to list commands and `leaf-hierarchy COMMAND --help`
to inspect a stage's options.

You can also use `python -m leaf_hierarchy` in place of `leaf-hierarchy`. On Windows,
use `.\.venv\Scripts\python.exe -m leaf_hierarchy` to run with the project's
environment without activating it first.

## Generated results

Starting with an empty `results/` directory, running `prepare`, `train` and `evaluate`
with the current PlantCLEF2015 data preparation and `resnet18_species` configuration
creates the structure below. Paths are relative to the repository root. The tree
shows the main preparation files and all training and evaluation files:

```text
results/
├── data/
│   └── plantclef2015/
│       └── v1/
│           ├── split_manifest.csv
│           ├── taxonomy.json
│           ├── config.json
│           └── ... additional dataset analysis files
└── runs/
    └── resnet18_species/
        └── RUN_ID/
            ├── config.json
            ├── history.csv
            ├── best.pt
            ├── validation/
            │   ├── validation_metrics.json
            │   ├── validation_species_metrics.json
            │   ├── validation_predictions.csv
            │   ├── validation_species_classification_report.csv
            │   └── validation_species_confusion_matrix.csv
            └── test/
                ├── evaluation_config.json
                ├── test_metrics.json
                ├── test_species_metrics.json
                ├── test_predictions.csv
                ├── test_species_classification_report.csv
                └── test_species_confusion_matrix.csv
```

`RUN_ID` is the date and time assigned when training starts, in
`YYYYMMDD_HHMMSS_microseconds` format. Training prints the resulting directory and
checkpoint path.

1. **Prepare data (`leaf-hierarchy prepare`).** Writes
   `results/data/plantclef2015/v1/`. The manifest records image paths, labels, groups,
   train/validation/test assignments and exclusions. The other files record taxonomy,
   preparation settings, inventories, species counts, duplicate checks and a
   distribution plot. Images stay in the configured dataset directory; preparation does
   not copy them. The [dataset guide](PlantCLEF2015TrainingData/README.md#prepare-the-dataset)
   lists the preparation outputs.

2. **Train (`leaf-hierarchy train`).** Creates a new run directory for the configured
   experiment. `config.json` records the settings and input identities; `history.csv`
   records training loss and validation metrics by epoch; `best.pt` contains the
   model selected using validation data. `validation/` saves metrics, reports and
   predictions for that selected checkpoint. Training does not create test results.
   Multitask runs also record the unweighted consistency loss, its weighted
   contribution and the coherence of independent validation predictions. Per-task
   losses remain cross-entropy. See the
   [consistency outputs](experiments/resnet18_consistency/README.md#saved-results)
   for the additional CSV columns and JSON fields.

3. **Evaluate (`leaf-hierarchy evaluate --checkpoint CHECKPOINT --split test`).**
   Writes `test/` beside the checkpoint; `--split validation` writes `validation/`
   instead. It saves evaluation settings, metrics, predictions, per-class reports
   and confusion matrices. Multitask checkpoints automatically include all three
   methods: `summary.csv` compares their metrics, and `independent/`,
   `species_path/` and `joint_path/` contain their reports and matrices.
   The original `SPLIT_*` files retain independent predictions and loss components.
   `SPLIT_metrics.json` also includes a `methods` object for the comparison.
   Saved logits, taxonomy and combined predictions allow the comparison to be
   inspected without another model pass. The
   [inference output layout](experiments/hierarchy_inference/README.md#outputs-and-interpretation)
   lists these files. Evaluation restores the checkpoint's consistency weight.

4. **Predict (`leaf-hierarchy predict --checkpoint CHECKPOINT --image IMAGE`).**
   Prints the prediction and softmax score in the terminal. It creates no result
   files; `--json` also prints to the terminal.

5. **Compare multiple checkpoints (`leaf-hierarchy compare-hierarchy --checkpoint CHECKPOINT --split validation`).**
   Creates a new directory under `results/analysis/hierarchy/` with results for
   all three inference methods. `summary.csv` contains each model's metrics;
   `aggregate.csv` contains means and sample standard deviations across models.
   Each model also has saved scores, per-image predictions, classification reports
   and confusion matrices. Use `--split test` to evaluate the same fixed rules on
   test. See the [experiment guide](experiments/hierarchy_inference/README.md#outputs-and-interpretation)
   for the complete output layout.

Reuse the prepared data across training runs: skip `prepare` when its output already
exists, because it refuses to overwrite that directory. Every `train` invocation
creates a new run directory. Repeating `evaluate` for the same checkpoint and split
replaces its generated files in that split's directory and keeps unrelated files.
An evaluation directory recorded for a model with different output tasks is rejected;
use a separate `--output-dir` for that model.

## Check the installation and shared code

From the repository root with the environment activated:

```text
python -m unittest discover -s tests -v
python -m pip check
```

Expect the test suite to end in `OK` and `No broken requirements found.` The tests use
synthetic inputs and need no dataset or trained model. They also exercise training,
checkpoint reload, evaluation and prediction for both model variants.

## Reproducibility

Prepared data, experiment definitions and individual runs have separate identities.
Keep the partition, taxonomy and preprocessing fixed when a comparison requires
them to be shared. Record any change to the data or evaluation protocol explicitly.

Raw dataset images, checkpoints and generated artifacts under `results/` are
excluded from Git. Temporary files under `tmp/` are also ignored. Preserve the
artifacts of evaluated runs separately, together with their configurations and
input fingerprints. Each experiment guide records its reproduction instructions
and evaluation results.
