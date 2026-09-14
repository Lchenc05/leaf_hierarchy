# leaf_hierarchy

Research code for an undergraduate thesis (TFG) on hierarchical plant classification
from leaf images. The thesis investigates the use of family, genus and species
relationships in classification and multitask learning, the taxonomic information
captured by learned representations, and the visual evidence behind model predictions.

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
  tests/                   # Model-independent checks for data, loss and metrics.
  report/                  # Thesis source and compilation instructions.
  pyproject.toml           # Package metadata, dependencies and command entry point.
```

The [experiment catalog](experiments/README.md) links each experiment's data
preparation guide, configuration, execution instructions and results.
The [thesis guide](report/README.md) covers writing and compiling the report.

## Installation

Use Git and 64-bit Python 3.11 or later. From a terminal:

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

Choose an experiment from the [catalog](experiments/README.md) and follow its dataset
guide to prepare the required data. Preparation produces a reusable manifest of
images, labels, groups and partition assignments. The experiment configuration
identifies that manifest and specifies the model and execution settings.

Run the classification pipeline from the repository root using the following command
templates. Replace `EXPERIMENT` with the chosen experiment directory, `CHECKPOINT`
with the saved model path printed by training, and `IMAGE` with an image path:

```text
leaf-hierarchy train --config experiments/EXPERIMENT/config.toml
leaf-hierarchy evaluate --checkpoint CHECKPOINT
leaf-hierarchy predict --checkpoint CHECKPOINT --image IMAGE
```

You can also use `python -m leaf_hierarchy` in place of `leaf-hierarchy`. On Windows,
use `.\.venv\Scripts\python.exe -m leaf_hierarchy` to run with the project's
environment without activating it first.

The [generated results](#generated-results) section below explains what each command
saves and where. Evaluation and prediction restore the model specification and
preprocessing from the checkpoint.

Data locations and manifest paths can be set in the configuration or supplied through
`--data-dir` and `--split-file`. Use the same prepared records for training and
evaluation. Each experiment guide provides complete commands with concrete paths.
Use `leaf-hierarchy --help` to list commands and `leaf-hierarchy COMMAND --help`
to inspect a stage's options.

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
            └── test/
                ├── evaluation_config.json
                ├── test_metrics.json
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
   distribution plot. Images stay in `PlantCLEF2015TrainingData/`; preparation does
   not copy them. The [dataset guide](PlantCLEF2015TrainingData/README.md#prepare-the-dataset)
   lists the preparation outputs.

2. **Train (`leaf-hierarchy train`).** Creates a new run directory for the configured
   experiment. `config.json` records the settings and input identities; `history.csv`
   records training loss and validation metrics by epoch; `best.pt` contains the
   model selected using validation data. Training does not create test results.

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
synthetic inputs and need no dataset or trained model. To check model execution as
well, follow the evaluation and prediction examples linked above.

## Reproducibility

Prepared data, experiment definitions and individual runs have separate identities.
Keep the partition, taxonomy and preprocessing fixed when a comparison requires
them to be shared. Record any change to the data or evaluation protocol explicitly.

Raw images, generated results and checkpoints are excluded from Git. Preserve the
artifacts of evaluated runs separately, together with their configurations and input
fingerprints. Each experiment guide records its reproduction instructions and
evaluation results.
