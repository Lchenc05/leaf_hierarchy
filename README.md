# leaf_hierarchy

Research code for hierarchical plant classification from leaf images. The project
investigates the use of family, genus and species relationships in classification
and multitask learning, the taxonomic information captured by learned
representations, and the visual evidence behind model predictions.

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
leaf-hierarchy evaluate --checkpoint CHECKPOINT
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

It saves metrics for each taxonomic level independently in validation and test.

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

3. **Evaluate (`leaf-hierarchy evaluate --checkpoint CHECKPOINT`).** Creates `test/`
   beside the selected checkpoint. It saves the evaluation settings, overall test
   metrics, predictions for each image, per-species precision/recall/F1 and the
   confusion matrix. Run this command separately after training.

4. **Predict (`leaf-hierarchy predict --checkpoint CHECKPOINT --image IMAGE`).**
   Prints the prediction and softmax score in the terminal. It creates no result
   files; `--json` also prints to the terminal.

Reuse the prepared data across training runs: skip `prepare` when its output already
exists, because it refuses to overwrite that directory. Every `train` invocation
creates a new run directory. Repeating `evaluate` for the same checkpoint replaces
its generated files in `test/`.

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

Raw images, generated results and checkpoints are excluded from Git. Preserve the
artifacts of evaluated runs separately, together with their configurations and input
fingerprints. Each experiment guide records its reproduction instructions and
evaluation results.
