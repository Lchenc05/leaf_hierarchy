# PlantCLEF2015 tree leaf scans

This directory documents the first dataset used by `leaf_hierarchy` and preserves
the botanical evidence for its tree-species selection. Dataset preparation produces
a shared manifest that model experiments can reuse.

## Obtain and locate the data

Download the dataset archive from the
[official PlantCLEF2015 training package](https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TrainingPackage/).
The project copy on
[Google Drive](https://drive.google.com/file/d/1SEEs5yGukc78jZ_hMylXVCYsC8O5YzDg/view?usp=drive_link)
is an alternative source; if access requires signing in, use an authorized account.
Allow space for the approximately 16 GB dataset archive, its extracted files,
dependencies and model checkpoints.

Extract the original JPG/XML pairs into `train/`:

```text
PlantCLEF2015TrainingData/
  README.md
  taxonomy_snapshot.json
  train/
    100373.jpg
    100373.xml
    ...
```

Some archives include an enclosing `PlantCLEF2015TrainingData/` directory. Place its
actual `train/` directory as shown above, retaining the repository's documentation
and frozen snapshot. Keep each image and its same-name XML. Use the training package
only: this protocol creates its own train/validation/test split within those data.

The [official task description](https://www.imageclef.org/lifeclef/2015/plant) reports
91,759 training images. The research copy used here has 91,758 image/XML pairs; the
missing pair is unknown. A different data copy can change counts or assignments.
Do not remove an arbitrary file to match the historical count. Retain the supplied
licence and attribution information; the official description reports Creative
Commons licensing.

Raw data and downloaded archives are ignored by Git. The guide and frozen botanical
snapshot are tracked. Additional datasets should have their own documentation and
raw-data ignore rules.

### Configure the dataset location

The extracted data can stay on another disk or outside the repository. Set the
directory containing `train/` in the existing `[data]` section of
[`experiments/resnet18_species/config.toml`](../experiments/resnet18_species/config.toml):

```toml
[data]
data_dir = "/opt/datasets/PlantCLEF2015TrainingData"
split_file = "results/data/plantclef2015/v1/split_manifest.csv"
```

On Windows, use a path such as `"D:/datasets/PlantCLEF2015TrainingData"`.
Use that configuration for preparation and training, then evaluate the saved
checkpoint (replace `CHECKPOINT` with the path printed by training):

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml
leaf-hierarchy train --config experiments/resnet18_species/config.toml
leaf-hierarchy evaluate --checkpoint CHECKPOINT
```

Evaluation reuses the data directory and manifest recorded in the checkpoint.
If you change the configured location later, also pass `--config` to evaluation.

Relative TOML paths start at the repository root. Keep the default `data_dir` value
when extracting the data inside the repository. See [advanced path options](#advanced-path-options)
for one-command overrides and a shared location across experiments.

Manifest image paths retain the canonical `PlantCLEF2015TrainingData/train/...`
prefix. The loader resolves it against the configured data directory; moving the
physical dataset does not change group identifiers. Do not edit those internal
paths to relocate data.
The guide and frozen `taxonomy_snapshot.json` stay in the repository; they do not
need to be copied alongside an external dataset.

## Prepare the dataset

After [installing the project](../README.md#installation), run from the repository root:

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml
```

Preparation reads and validates only `[data]` from this file. `data_dir` locates the
images; `split_file` sets the manifest path and the directory for all preparation
outputs. The model and training settings do not affect preparation.
You can omit `--config` to use the default data and output locations.

The preparation workflow reads and audits XML metadata, selects tree-species
LeafScan images, checks image decoding, identifies observations and duplicate
groups, summarizes the selected data and allocates groups to splits. It does not
modify the raw images or XML. Full preparation decodes images and compares duplicate
candidates, so it can take several minutes.

The botanical selection uses 351 decisions and 624 source extracts from BGCI
GlobalTreeSearch 1.9 and Kew WCVP 16, consulted on 9 September 2026. The
[frozen snapshot](taxonomy_snapshot.json) preserves those records, their citations,
versions, source identifiers and source-archive hashes. Preparation validates the
recorded decisions against the source extracts without downloading live taxonomy.
Original evidence is retained without translation; generated artifact names,
column names and operational codes use English.

The saved protocol uses split seed `20260909` and targets 70%/15%/15% by complete
groups within each species. A species needs at least 20 usable groups, including at
least 14 training groups and three groups in each evaluation split. Groups with
missing observation identifiers or conflicting labels are excluded. These are the
initial comparison protocol's choices, rather than requirements imposed by the
PlantCLEF2015 dataset itself.

With the reference configuration, outputs go to `results/data/plantclef2015/v1/`:

| Artifact | Contents |
| --- | --- |
| `inventory.csv` | Image/XML inventory and metadata diagnostics |
| `tree_leaf_scans.csv` | Selected readable tree leaf scans and their groups |
| `split_manifest.csv` | Per-image labels, group, split and exclusion reason |
| `exclusions.csv` | Filtering and split exclusions |
| `tree_species_criteria.csv` | Botanical selection decisions in the output schema |
| `taxonomy_source_extracts.csv`, `taxonomy_source_provenance.csv` | Preserved source records and provenance |
| `taxonomy.json` | Class mappings and parent relationships from the active dataset labels |
| `manifest_schema.json` | Manifest schema, provenance and fingerprints |
| `config.json` | Seed, thresholds, source-snapshot hash and software versions |
| `split_checks.csv`, `split_counts.csv`, `split_totals.csv` | Separation checks and split support |
| `duplicate_audit.csv`, `group_audit.csv`, `species_eligibility.csv` | Duplicate evidence, quarantines and class eligibility |
| `species_observation_distribution.png` | Distribution of selected images and observations |

Additional XML, inventory and distribution diagnostics accompany these outputs.
The manifest's shared fields include `image_path`, `species`, `genus`, `family`,
`class_id`, `observation_id`, `group_id`, `split` and `content`.

Preserved historical outputs live in `results/leafscan/`, with English filenames
including `inventory.csv`, `tree_leaf_scans.csv`, `split_manifest.csv` and
`exclusions.csv`. That historical manifest remains separate from freshly prepared
versions under `results/data/plantclef2015/`. Use the manifest associated with a
saved checkpoint when reproducing its evaluation.

Preparation publishes its outputs after completing its checks and refuses to
overwrite an existing output directory. Reuse an existing manifest for additional
training runs. To prepare a new version, change `[data].split_file` to a path in a
new directory, for example `results/data/plantclef2015/v2/split_manifest.csv`, then
run `prepare --config` again. Training with the same TOML will read that version.
Inspect failures before proceeding; do not disable the split or source-evidence
checks. `leaf-hierarchy prepare --help` also documents `--workers` and
`--taxonomy-file` for an explicitly selected source snapshot.

## Advanced path options

### Override a location for one command

Use `--data-dir` to override the configured raw dataset location:

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml --data-dir "D:/datasets/PlantCLEF2015TrainingData"
```

`--output-dir` overrides the directory for preparation outputs while preserving
the manifest filename from `[data].split_file`. For example, if `split_file` is
`results/data/plantclef2015/v2/records.csv`, this command writes
`results/data/plantclef2015/custom/records.csv` and its supporting files:

```text
leaf-hierarchy prepare --config experiments/resnet18_species/config.toml --output-dir results/data/plantclef2015/custom
```

Without a configuration, the manifest filename is `split_manifest.csv`. After an
output-directory override, update `[data].split_file` before training or supply
the new manifest path with `train --split-file`. Evaluation reuses the paths saved
by training; `evaluate --config` reads an updated TOML, and `--split-file` overrides
the configured or recorded manifest for one command.

### Share a dataset location across experiments

Set `LEAF_HIERARCHY_DATA_DIR` once for commands in the current terminal:

```bash
export LEAF_HIERARCHY_DATA_DIR=/opt/datasets/PlantCLEF2015TrainingData
```

```powershell
$env:LEAF_HIERARCHY_DATA_DIR = "D:/datasets/PlantCLEF2015TrainingData"
```

The variable applies to `prepare`, `train` and `evaluate`, including evaluation of
a checkpoint saved on another machine. Add it to your shell profile to reuse the
setting in future terminals. Prediction takes the image's actual path with `--image`.

The priority is `--data-dir` > `LEAF_HIERARCHY_DATA_DIR` > `[data].data_dir` >
the repository's `PlantCLEF2015TrainingData/` directory. Evaluation without
`--config` uses the checkpoint's saved path before the repository default.
Relative environment and command paths start at the current directory. All path
sources support `~` for the home directory. Empty or whitespace-only environment
values are ignored.

## Taxonomic labels

The initial protocol preserves `Family`, `Genus` and `Species` values from the
original XML, exposing them as lowercase fields in the shared manifest. The frozen
botanical sources determine tree-species inclusion. They do not silently replace
the XML's family/genus hierarchy with another taxonomy.

The original selection has 9,648 images, 120 species, 67 genera and 36 families.
The active baseline partition has 4,695 images, 41 species, 32 genera and 21 families.
In that saved partition, species-to-genus and genus-to-family relationships are
unique, and all three label levels are present for every active record.

There is a documented difference between authorities: the XML assigns
`Viburnum lantana L.` and `Viburnum tinus L.` to `Adoxaceae`, while their frozen WCVP
accepted records use `Viburnaceae`. The original labels are retained. Before an
experiment uses a harmonized taxonomy, choose its authority and version, record
the old-to-new correspondence and save a distinct hierarchy version. This decision
belongs to the research protocol.

The snapshot retains its original bytes. Split seeds incorporate the original species
names, so changing taxonomic labels can also change partition assignments. Treat such
changes as a separate data version.

See the [ResNet18 experiment](../experiments/resnet18_species/README.md) for the
historical split counts, measurements and limitations, including instructions for
[reusing existing artifacts](../experiments/resnet18_species/README.md#existing-artifacts).

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing images or XML | Check the extracted `train/` directory and the data location in `--config`, `LEAF_HIERARCHY_DATA_DIR` or `--data-dir`. |
| Output directory already exists | Reuse its manifest or set `[data].split_file` to a path in a new directory. |
| Botanical snapshot check fails | Check the data edition and snapshot integrity; review changed species labels against the recorded sources. |
| Counts differ from the reference | Inspect the dataset inventory and exclusions before changing the selection or files. |
