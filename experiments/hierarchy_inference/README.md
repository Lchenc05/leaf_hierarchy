# Hierarchical inference on saved multitask models

## Research question

How often do the independent family, genus and species predictions contradict the
dataset taxonomy? Can a valid-path decision rule remove those contradictions
without reducing classification accuracy or macro F1?

This experiment compares **three inference methods on each saved multitask
checkpoint**. `independent` is the original multitask prediction, not the separate
species-only ResNet18 model. The comparison reuses the saved model parameters.

The comparison is available through `compare-hierarchy`. The `evaluate` and
`predict` commands use independent output heads.

## Fixed protocol

1. **Independent**: take the largest logit separately in each output head.
2. **Species path**: take the species head's prediction and look up its genus and
   family in the checkpoint taxonomy. The species prediction stays unchanged.
3. **Joint path**: enumerate the valid paths represented by the checkpoint's
   species vocabulary. For a candidate species `s`, let `g(s)` and `f(s)` be its
   saved genus and family. Maximize:

   ```text
   score(s) = log p_species(s | image)
            + log p_genus(g(s) | image)
            + log p_family(f(s) | image)
   ```

The probabilities are head-wise softmax scores. All three terms have fixed weight
1 and temperature 1, specified before the initial validation/test comparison below.
Log-space computation avoids numerical underflow. Exact score ties select the
lowest saved species index. This product-of-heads rule is a scoring heuristic,
not a calibrated joint probability: the heads share features and are not assumed
to be independent random observations.

All three methods use the **same logits from one image pass**. The species-derived
and joint methods only return complete paths present in the checkpoint taxonomy;
independent predictions can contradict those relationships. Relations are not
inferred from species name spelling. The taxonomy comes from dataset labels, not
a live botanical lookup.

The command uses the fixed weights above. `--split` selects the images to evaluate.

The checkpoint's preprocessing, class order and dataset partition are restored and
verified. Multiple checkpoints in one comparison must share their partition,
taxonomy and preprocessing. The same fixed decoding rules are applied to
validation and test.

## Run the comparison

First [install the project](../../README.md#installation) and make the multitask
checkpoints, their original split manifest and the dataset available locally.
These artifacts are excluded from Git and must be supplied separately. The
checkpoint must contain family, genus and species heads and their taxonomy.

From the repository root, with the environment activated:

```text
leaf-hierarchy compare-hierarchy --checkpoint CHECKPOINT --split validation
leaf-hierarchy compare-hierarchy --checkpoint CHECKPOINT --split test
```

Replace `CHECKPOINT` with the path to a multitask `best.pt`, absolute or relative
to the current directory. Quote paths containing spaces. Validation is the default
split. You can compare multiple checkpoints in one invocation:

```text
leaf-hierarchy compare-hierarchy --checkpoint CHECKPOINT_SEED_1 CHECKPOINT_SEED_2 CHECKPOINT_SEED_3 --split validation
```

To reproduce the initial three-model comparison on Windows without activating
the environment, use the saved checkpoints at these local paths:

```powershell
.\.venv\Scripts\python.exe -m leaf_hierarchy compare-hierarchy --checkpoint results/runs/resnet18_multitask/20260917_190833_752991/best.pt results/runs/resnet18_multitask/20260924_145031_109112/best.pt results/runs/resnet18_multitask/20260924_192012_081762/best.pt --split validation
```

Repeat the command with `--split test` once the rule is fixed. Reuse the prepared
manifest and trained models. Without `--config`, dataset paths and batch size come
from each checkpoint when recorded. `LEAF_HIERARCHY_DATA_DIR` overrides the saved
dataset directory, and explicit command-line flags take precedence. Use
`--data-dir` or `--split-file` if those files have moved.

An explicit `--config` TOML supplies data/runtime/batch settings; the dataset
environment variable still overrides its dataset directory, and command-line
flags override both. `--batch-size`, `--device` and `--num-threads` are available.
Without overrides, runtime defaults are device `auto` and six CPU threads.
The architecture, taxonomy and preprocessing come from the checkpoint.

## Outputs and interpretation

Each command creates a new directory under `results/analysis/hierarchy/`.
`--output-dir PATH` can name a new output directory. Existing output directories
are rejected so a previous comparison and the original `test/` results stay intact.
`comparison_config.json` is marked `complete` only after all checkpoints finish;
an interrupted or failed invocation must not be treated as a finished comparison.

```text
COMPARISON/
  comparison_config.json  # Protocol, input paths, source hashes and completion state.
  summary.csv             # Per-checkpoint, method and taxonomic-level results.
  aggregate.csv           # Mean and sample standard deviation across checkpoints.
  01_seed_SEED/
    config.json           # Checkpoint hash, epoch, seed, input identities and runtime.
    taxonomy.json
    scores.npz            # Logits, log-softmax scores, class names and ordered image IDs.
    predictions.csv       # True labels, each method's labels and consistency flags.
    metrics.json
    independent/          # Classification reports and confusion matrices for F/G/S.
    species_path/
    joint_path/
```

Scores in `scores.npz` can be read with `numpy.load(..., allow_pickle=False)`.
The arrays contain the original logits, log-probabilities, class names and true
labels. Their per-image order matches `predictions.csv`.

- **Coherence rate**: fraction of predicted triplets whose species belongs to the
  predicted genus and whose genus belongs to the predicted family.
- **Pair consistency**: species-genus, genus-family and species-family compatibility,
  recorded separately in `metrics.json` and per-image flags.
- **Accuracy**: fraction of images classified correctly at each taxonomic level.
- **Macro F1**: mean of the per-class F1 scores at each level, using the complete
  checkpoint vocabulary, including classes absent from that split. Undefined
  per-class scores are set to zero.
- **Complete-path accuracy**: fraction of images for which all three labels are
  correct simultaneously.
- **Changed/fixed/harmed**: counts relative to independent decoding for each level.
  A fixed prediction was incorrect and becomes correct; a harmed one does the reverse.

Species-derived and joint predictions are 100% coherent by construction, but they
can be consistently wrong. Interpret that guarantee together with accuracy/F1 and
the fixed/harmed counts. Species-derived decoding cannot improve species accuracy.
When the independent argmax triplet is already valid, the equal-weight joint rule
keeps its optimal path (apart from score ties).

`aggregate.csv` reports sample standard deviations (`ddof=1`); with only one
checkpoint the SD cell is empty. These are repetitions on the same held-out images,
not additional independent test samples.

## Initial measured comparison (26 September 2026)

The fixed protocol above was applied to the three multitask checkpoints with
training seeds `20260910`, `20260924` and `20260927`. It was first run on validation
(643 images), then unchanged on test (683 images). No temperature or task weight
was selected using either set. All original independent predictions were reproduced
image by image, and all saved scores were decoded again to verify the exports.

Test species metrics, mean ± sample SD across three checkpoints, in percent:

| Inference | Coherence | Species accuracy | Species macro F1 |
| --- | --- | --- | --- |
| Independent | 96.29 ± 0.68 | 88.58 ± 2.16 | 88.81 ± 2.51 |
| Species path | 100.00 ± 0.00 | 88.58 ± 2.16 | 88.81 ± 2.51 |
| Joint path | 100.00 ± 0.00 | 88.97 ± 2.64 | 89.26 ± 3.11 |

The independent predictions have 28, 28 and 20 invalid paths, respectively, on
the same 683 test images. Joint decoding improves mean species accuracy by 0.39
percentage points and macro F1 by 0.44 points, calculated before rounding. The
changes vary across seeds: accuracy improves in the first and third, and is
unchanged in the second; macro F1 decreases slightly in the second. These three
repetitions do not establish statistical superiority.

On validation, independent vs joint mean species accuracy is 86.26% vs 86.47%,
and macro F1 is 85.40% vs 85.42%. Coherence increases from 93.83% to 100%.

The complete local outputs are in
`results/analysis/hierarchy/20260926_validation_v1/` and
`results/analysis/hierarchy/20260926_test_v1/`. They include family and genus
metrics, per-image fixes/regressions, score arrays and checkpoint/data identities.
The species-only ResNet18 experiment is not one of these three decoding methods.

A separate exploratory analysis on 28 September varied family and genus weights
using the saved validation scores. The command's fixed `1:1:1` rule and the initial
measurements above remain unchanged.

## Verification

```text
python -m unittest discover -s tests -v
```

Checks include a hand-calculated joint path, non-alphabetical class indices,
deterministic ties, extreme logits, false improvements/regressions despite valid
paths, and a train-save-load workflow that reproduces the original independent
predictions on validation and test. Saved score files are decoded again to check
that the exported joint predictions are reproducible.
