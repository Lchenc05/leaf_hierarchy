# Experiment catalog

Each experiment has a directory containing its research question, configuration,
execution instructions, evaluation protocol and measured results. Shared command
examples and installation instructions are in the [project README](../README.md).

| Experiment | Purpose | Data | Configuration |
| --- | --- | --- | --- |
| [ResNet18 species classification](resnet18_species/README.md) | Species-classification reference for comparisons of taxonomic learning objectives. | [PlantCLEF2015 tree leaf scans](../PlantCLEF2015TrainingData/README.md) | [config.toml](resnet18_species/config.toml) |
| [ResNet18 multitask classification](resnet18_multitask/README.md) | Shared backbone with family, genus and species heads; separate validation and test metrics. | Same prepared manifest as the species baseline | [config.toml](resnet18_multitask/config.toml) |

## Run the reference experiment

After [installing the project](../README.md#installation), download and extract the
dataset from the repository root using Bash (for example, on Linux or in WSL):

```bash
wget -c "https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TrainingPackage/PlantCLEF2015TrainingData.tar.gz"
mkdir -p PlantCLEF2015TrainingData
tar -xzf PlantCLEF2015TrainingData.tar.gz -C PlantCLEF2015TrainingData
```

Check the extracted directory layout against the
[dataset guide](../PlantCLEF2015TrainingData/README.md#obtain-and-locate-the-data).
Run the following commands from the repository root with the environment activated.
Set `[data].data_dir` in the experiment's
[config.toml](resnet18_species/config.toml) to the directory containing `train/`.

Prepare the data once. Skip this step if
`results/data/plantclef2015/v1/split_manifest.csv` already exists:

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml
```

Preparation uses `[data].split_file` for the manifest path and writes its supporting
files in the same directory, which must not already exist. It reads only `[data]`
from the configuration.

Train the five-epoch reference configuration:

```text
leaf-hierarchy train --config experiments/resnet18_species/config.toml
```

Training prints the run directory and the path to `best.pt`. In both commands
below, replace `RUN_ID` with that run directory's name, or replace the whole
checkpoint path with the one printed by training.

Evaluate the selected model on the held-out test split:

```text
leaf-hierarchy evaluate --checkpoint results/runs/resnet18_species/RUN_ID/best.pt
```

The command prints accuracy and macro F1 and saves metrics, per-species reports,
a confusion matrix and per-image predictions in that run's `test/` directory.

Predict the species of one image:

```text
leaf-hierarchy predict --checkpoint results/runs/resnet18_species/RUN_ID/best.pt --image IMAGE
```

Prediction prints a species name and a softmax score. Replace `IMAGE` with the
actual image path; add `--json` for structured output. A single
prediction checks inference; use test evaluation to measure model performance.

## Check the saved historical model

If you have the original checkpoint, manifest and dataset, you can evaluate and
predict directly without training again. These files are excluded from Git and
must be supplied separately. With the historical files in their documented local
locations, run from the repository root:

```text
leaf-hierarchy evaluate --checkpoint results/resnet18/20260911_130918_664606/resnet18.pt --split-file results/leafscan/split_manifest.csv --device cpu
leaf-hierarchy predict --checkpoint results/resnet18/20260911_130918_664606/resnet18.pt --image PlantCLEF2015TrainingData/train/100373.jpg --device cpu
```

Evaluation results are saved in `results/resnet18/20260911_130918_664606/test/`.
The preserved historical manifest is `results/leafscan/split_manifest.csv`.
Fresh preparation writes a separate version under `results/data/plantclef2015/v1/`.
Keep each run associated with the manifest used for its training.

The CPU check on 14 September 2026 evaluated 683 test images from 41 species and
reproduced accuracy `0.8697` and macro F1 `0.8834`, using epoch 5. The example image
returned `Prunus spinosa L.` with softmax score `0.8699`, matching its XML label.
These values describe this historical checkpoint; a new training run has its own
measurements. The score is not a calibrated probability of correctness.

The historical checkpoint emits a warning because it lacks the original partition
fingerprint. Class mappings and group separation are checked, but the original
training partition cannot be verified automatically. See the
[experiment guide](resnet18_species/README.md#existing-artifacts) for data-location
options and historical compatibility details.

## Run the shared checks

After installation, these checks require no dataset or trained model:

```text
python -m unittest discover -s tests -v
python -m pip check
```

Expect the test suite to end in `OK` and `No broken requirements found.` These checks
cover data validation, taxonomy, loss and metrics. The evaluation and prediction
commands above exercise the saved model. The [experiment guide](resnet18_species/README.md)
records the full environment, protocol and limitations.

## Documenting an experiment

Add a directory and a catalog entry for each distinct experimental protocol. Its
documentation should identify the question being tested, dataset and taxonomy
versions, configuration, selection rule, measurements and limitations. Place reusable
implementation in `src/leaf_hierarchy/` and keep generated runs under `results/`.

Multiple runs can share an experiment definition. Record their resolved settings
and input fingerprints so differences in initialization or configuration remain
traceable. Document completed measurements with the corresponding run identity.
