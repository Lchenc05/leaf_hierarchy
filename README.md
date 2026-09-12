# leaf_hierarchy

Reproducible analysis of PlantCLEF2015 tree leaf scans and a ResNet18 species-classification
baseline. All project code runs from Python files; no notebook or Jupyter installation
is required. The current model has one species head. Family/genus/species multitask
learning remains a subsequent experiment.

The experiment uses a frozen botanical selection and a grouped train/validation/test split.

## Project files

```text
leaf_hierarchy/
  analyze_data.py
  train.py
  evaluate.py
  predict.py
  model_utils.py
  taxonomy_snapshot.json
  requirements.txt
  README.md
  .gitignore
  PlantCLEF2015TrainingData/
    README.md
    train/                    # Downloaded JPG/XML pairs; ignored by Git.
  results/                    # Generated analysis and experiments; ignored by Git.
  report/
    main.tex
    metadata.tex
    english_layout.tex
    build.ps1                 # Windows build helper for MiKTeX.
    README.md
    references.bib
    sections/
    figures/
    TeXiS/                    # Official UCM template support files.
```

`model_utils.py` shares model construction, preprocessing, loading and validation.
`taxonomy_snapshot.json` preserves the original botanical decisions and source evidence.
The legacy notebook, if present locally, is archived under the ignored `archive/` folder.
It is not needed by any script. `requirements-baseline.txt` forwards to `requirements.txt`
for compatibility with earlier installation commands.

## 1. Install dependencies

Use Git and **64-bit Python 3.13**. The historical experiment used Python 3.13.9 on
Windows with CPU-only PyTorch. A GPU is optional. Allow space for the approximately
16 GB dataset archive, its extracted contents, dependencies and model checkpoints.

```text
git clone https://github.com/Lchenc05/leaf_hierarchy.git
cd leaf_hierarchy
```

Create an isolated environment.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

For CPU training on Windows or Linux:

```text
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

On macOS, omit the command containing `--index-url`; the requirements file installs
the platform wheels. For an NVIDIA GPU, substitute the matching CUDA wheel command
for **PyTorch 2.10.0 / torchvision 0.25.0** from the
[official instructions](https://pytorch.org/get-started/previous-versions/#v2100),
then install `requirements.txt`. CUDA also needs a compatible NVIDIA driver.
`--device auto` selects CUDA if available and CPU otherwise; MPS is not selected.

If PowerShell blocks environment activation, replace `python` with
`.\.venv\Scripts\python.exe` in subsequent commands. Verify the environment:

```text
python -c "import torch, torchvision, numpy, pandas, scipy, PIL, sklearn, matplotlib; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 2. Download and place the dataset

Download the dataset archive shared for this project from
[Google Drive](https://drive.google.com/file/d/1SEEs5yGukc78jZ_hMylXVCYsC8O5YzDg/view?usp=drive_link) using the download button.
If Drive requests sign-in or access, use an authorized Google account or the official
source below.
The [official PlantCLEF2015 training package](https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TrainingPackage/)
is also available as an alternative source.

Extract its `train/` folder into `PlantCLEF2015TrainingData/` beside the Python scripts.
The shared archive may already contain the outer `PlantCLEF2015TrainingData/`
directory. Copy its `train/` folder into the existing dataset directory; keep the
repository's README files when the archive contains its own copies.
Keep both each JPG and its same-name XML; for example:

```text
PlantCLEF2015TrainingData/train/100373.jpg
PlantCLEF2015TrainingData/train/100373.xml
```

If extraction adds another enclosing directory, place the actual `train/` folder at
the location above. Do not mix the official test package into this directory: the
experiment makes its own train/validation/test split within the official training data.

The [official task description](https://www.imageclef.org/lifeclef/2015/plant) reports
91,759 training images and Creative Commons licensing. The research copy used here has
91,758 image/XML pairs; the missing pair is unknown. Keep the supplied licence and
attribution information and check it before redistribution. A different data copy can
produce different counts or assignments; do not delete an arbitrary image to match them.

Each dataset has its own directory. Git tracks only the placement guide in
`PlantCLEF2015TrainingData/`, not its raw files. Use a separate directory and ignore
rule if adding another dataset.

**Existing data need not be copied into the repository.** Every data-dependent command
accepts `--data-dir` pointing to the folder containing `train/`. For example:

```text
python analyze_data.py --data-dir "D:/datasets/PlantCLEF2015TrainingData"
python train.py --data-dir "D:/datasets/PlantCLEF2015TrainingData"
```

Use the same `--data-dir` for evaluation. Manifest paths remain canonical
`PlantCLEF2015TrainingData/train/...` paths, so moving the dataset directory does not
change duplicate-group identifiers. `TFG_ROOT` and `TFG_OUTPUT_DIR` are no longer used;
paths are supplied through command-line options. Default paths are relative to the
Python files, and explicitly supplied relative paths are relative to your terminal.

## 3. Analyze the raw data

From the repository root, after placing the data as described:

```text
python analyze_data.py
```

The command reads the XML metadata, applies the frozen tree-species selection, checks
image decoding, detects related/duplicate images, produces distribution summaries and
assigns groups to splits. It does not modify the source images/XML.

The selection uses 351 decisions and 624 source excerpts from BGCI GlobalTreeSearch 1.9
and Kew WCVP 16, consulted on 9 September 2026, preserved in `taxonomy_snapshot.json`.
No notebook or live taxonomy download is required. Source quotations, scientific names
and legacy CSV keys and taxonomic decision codes retain their original form. Python comments and
identifiers are in English.

Outputs are written to `results/analysis/`:

| File | Contents |
| --- | --- |
| `inventario_completo.csv` | Complete image/XML inventory and diagnostics. |
| `inventario_leafscan_arboles.csv` | Readable tree-species leaf scans and groups. |
| `particion_propuesta.csv` | Per-image train/validation/test/excluded assignment and reasons. |
| `exclusiones.csv` | Filtering and split exclusions. |
| `config.json` | Seeds, thresholds, source-snapshot hash and software versions. |
| `split_checks.csv`, `split_counts.csv`, `split_totals.csv` | Separation checks and split support. |
| `duplicate_audit.csv`, `group_audit.csv`, `species_eligibility.csv` | Duplicate evidence, quarantines and eligible classes. |
| `species_observation_distribution.png` | Distribution plot. |

Additional source and diagnostic tables are saved in the same directory. Rerunning
analysis replaces files with matching names there. Use another output directory to
preserve a run:

```text
python analyze_data.py --output-dir results/analysis_v2 --workers 8
```

When changing that directory, pass its manifest to both training and evaluation with
`--split-file results/analysis_v2/particion_propuesta.csv`. The analysis prints its
progress; full XML processing, image decoding and duplicate checking can take several
minutes. Address errors before continuing and do not disable its assertions.

## 4. Train the baseline

With the generated default manifest and local dataset:

```text
python train.py
```

ImageNet weights download on the first run unless cached. `TORCH_HOME` can specify an
existing cache; otherwise `.cache/torch/` is used. Training prints the new run directory
and the path of the selected `resnet18.pt` checkpoint. It writes:

- `config.json`: hyperparameters, class mapping, split sizes and hashes, software versions.
- `history.csv`: train loss and validation loss/accuracy/macro F1 after each epoch.
- `resnet18.pt`: the best validation checkpoint, selected epoch and class mapping.

These files are inside `results/resnet18/<run>/`. The run name is a timestamp and must
be copied from the command output for the next step. Test data are not used to select
the model. Interrupted training can leave a partial history/checkpoint; automatic resume
is not implemented, so start another run for a complete experiment.

| Setting | Default |
| --- | --- |
| Split seed | `20260909` |
| Split allocation | 70% / 15% / 15% by groups within species |
| Minimum support | 20 usable groups/species: at least 14 train, 3 validation and 3 test |
| Model | ResNet18 with ImageNet `IMAGENET1K_V1`, one output per admitted species |
| Training seed | `20260910` |
| Epochs / batch size | 5 / 16 |
| Optimizer | AdamW, all parameters trained |
| Learning rate / weight decay | `0.0001` / `0.0001` |
| Selection | Highest validation macro F1; lower loss breaks ties |

Preprocessing handles EXIF orientation, converts to RGB, resizes to 256, crops to 224
and normalizes with ImageNet statistics. Training additionally uses independent
horizontal/vertical flips with probability 0.5 each. Validation and test use no random
augmentation. Random state is reset immediately before model creation.

To change runtime settings, for example:

```text
python train.py --epochs 5 --batch-size 16 --device cpu --output-dir results/resnet18
```

`--weights none` allows an offline functional check, but changes initialization and
does not reproduce the ImageNet baseline. A shorter run is also not a substitute for
the complete five-epoch experiment.

## 5. Evaluate the selected model

Replace `RUN_ID` with the run directory name printed by training:

```text
python evaluate.py --checkpoint results/resnet18/RUN_ID/resnet18.pt
```

The command loads the selected checkpoint and evaluates only the held-out test split.
It saves these files under `results/resnet18/<run>/test/`:

| File | Contents |
| --- | --- |
| `test_metrics.json` | Selected epoch, test loss, accuracy, macro F1 and support. |
| `test_classification_report.csv` | Per-species precision, recall, F1 and support. |
| `test_confusion_matrix.csv` | True species in rows, predicted species in columns. |
| `test_predictions.csv` | Image, true species, group and predicted species. |

Accuracy and macro F1 are image-level metrics; macro F1 gives each species equal
weight. Grouped splitting does not make them group-weighted metrics.

Use the same manifest used for training. The checkpoint stores an active-record
fingerprint, so new checkpoints reject changed split assignments. The historical
notebook checkpoints are supported; they lack this fingerprint and emit a warning
because automatic verification of their original partition is unavailable. Class
mapping and group separation are still checked.

For a custom data location, split and output directory:

```text
python evaluate.py --checkpoint results/resnet18/RUN_ID/resnet18.pt --data-dir "D:/datasets/PlantCLEF2015TrainingData" --split-file results/analysis_v2/particion_propuesta.csv --output-dir results/evaluation_v2 --device cpu
```

Keep the full active manifest and its referenced images available for these checks,
including train and validation records. Evaluation results can be regenerated without
retraining. Avoid repeatedly using test results to choose model configurations.

### Reuse existing local analyses and checkpoints

Earlier experiments used `resultados_leafscan/` for analysis CSVs and
`resultados_resnet18/` for checkpoints. These optional local folders are ignored by
Git and are not included in a fresh clone. New analysis writes to `results/analysis/`,
and new training writes to `results/resnet18/`; the earlier folders are preserved.

If you already have `resultados_leafscan/particion_propuesta.csv`, you can train
directly from it without repeating the analysis:

```text
python train.py --split-file resultados_leafscan/particion_propuesta.csv
```

Evaluate the resulting run using the same saved manifest:

```text
python evaluate.py --checkpoint results/resnet18/RUN_ID/resnet18.pt --split-file resultados_leafscan/particion_propuesta.csv
```

To evaluate the original reference checkpoint, if you have kept its local files:

```text
python evaluate.py --checkpoint resultados_resnet18/20260911_130918_664606/resnet18.pt --split-file resultados_leafscan/particion_propuesta.csv --output-dir results/historical_test
```

Supply `--data-dir` as well when your dataset is stored elsewhere. A missing default
`results/analysis/particion_propuesta.csv` does not mean the earlier analysis was lost;
pass its existing location with `--split-file`.

## 6. Predict an image

```text
python predict.py --checkpoint results/resnet18/RUN_ID/resnet18.pt --image PlantCLEF2015TrainingData/train/100373.jpg
```

Any existing image path can replace the example. This command needs only the image,
checkpoint and model dependencies; no dataset manifest or XML files are required.
It prints a species and its softmax score. The score is not a calibrated probability
of correctness, and the classifier always chooses among its trained classes.
Add `--json` for a machine-readable result.

See all supported options with `python analyze_data.py --help`, `python train.py --help`,
`python evaluate.py --help` and `python predict.py --help`.

## 7. Write and compile the thesis

`report/` contains an empty thesis template based on the official
[UCM 2026/2027 LaTeX template](https://informatica.ucm.es/tfgs-2026-2027), with `sections/` and `figures/`.
It keeps the cover and chapter headings, with empty English and Spanish abstract
and keyword sections. Edit `report/metadata.tex`
to set the titles, author, supervisor, degree and submission details. Optional empty
collaborator, date and submission-period fields are omitted from the cover.

On Windows, install [MiKTeX](https://miktex.org/download) for your user and enable
automatic installation of missing packages in MiKTeX Console. Install the LaTeX
Workshop extension in VS Code and open the repository folder. Its default
**Build thesis with MiKTeX** recipe uses the included PowerShell helper.
Saving `report/metadata.tex` or a chapter builds `report/main.tex`.

To build from the repository root in Windows PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\report\build.ps1
```

The execution-policy option applies only to this command. The helper finds MiKTeX
even when VS Code has not yet refreshed its PATH. The first build needs internet
access to obtain missing LaTeX packages. It uses MiKTeX's `texify`, so Perl and
`latexmk` are not required for this Windows recipe.

On Linux/macOS, install TeX Live or MacTeX with `latexmk`, then use:

```text
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Select **Build thesis with latexmk** in LaTeX Workshop on those systems.
This produces `report/main.pdf`; generated LaTeX files are ignored. See
`report/README.md` for manual commands, official-template provenance and requirements.
The chapter bodies and bibliography are intentionally empty. Add the thesis text
when you are ready to write it; the project results remain documented separately below.

## Reference experiment and limits

The original local analysis selected 9,648 images from 120 species. The active baseline
used 4,695 images from 41 species:

| Split | Images | Groups | Species |
| --- | ---: | ---: | ---: |
| Train | 3,369 | 1,263 | 41 |
| Validation | 643 | 271 | 41 |
| Test | 683 | 271 | 41 |

Historical run `20260911_130918_664606` selected epoch 5 and reported validation accuracy
0.8616, validation macro F1 0.8546, test accuracy **0.8697** and test macro F1 **0.8834**.
These values are a reference, not a guarantee for another data copy or environment.
Use the saved files from your own completed run for its measurements.

Verification has included a full raw-data analysis reproducing the original
assignments, a short training/checkpoint/evaluation round trip, and evaluation of
the reference checkpoint. Direct dependencies are pinned; hardware and platform
libraries can still affect exact scores.

Classes are imbalanced and 35 of 41 species have fewer than ten test groups. Splitting
keeps observations and detected duplicate groups together, but cannot guarantee different
physical trees across observations. Duplicate detection can miss transformed images or
group false positives. Results are exploratory and limited to the selected leaf scans;
field photographs, unseen species and the multitask architecture remain unevaluated.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing Python module | Activate `.venv` and install `requirements.txt` with that interpreter. |
| Missing dataset/XML | Check `PlantCLEF2015TrainingData/train/` and `--data-dir`. |
| Missing split manifest | Run `analyze_data.py`, then use its path with `--split-file`. |
| Botanical snapshot assertion | Check that the 2015 training package and frozen JSON are intact; check new species labels against the recorded botanical sources. |
| Checkpoint/split mismatch | Use the exact active split used for training; do not edit assignments to bypass the check. |
| CUDA out of memory | Reduce `--batch-size` or use `--device cpu`; record the changed settings. |
| ImageNet download fails | Restore internet access or provide its weights in `TORCH_HOME`. |
| LaTeX command not found | Install a LaTeX distribution; see `report/README.md`. |

Raw datasets, generated results, large checkpoints, caches, environments and the local
notebook archive are ignored. Keep the selected report figures, source files and frozen
taxonomy evidence under version control. Archive the evaluated run artifacts separately
when sharing the final research results.
