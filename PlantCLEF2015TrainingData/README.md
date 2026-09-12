# PlantCLEF2015 training data

Download the dataset archive shared for this project from
[Google Drive](https://drive.google.com/file/d/1SEEs5yGukc78jZ_hMylXVCYsC8O5YzDg/view?usp=drive_link) using the download button.
If Drive requests sign-in or access, use an authorized Google account or the official
source below.
The [official PlantCLEF2015 training package](https://lab.plantnet.org/LifeCLEF/PlantCLEF2015/TrainingPackage/)
is available as an alternative source.

Extract the original JPG/XML pairs into `train/` inside this directory:

```text
PlantCLEF2015TrainingData/
  README.md
  train/
    100373.jpg
    100373.xml
    ...
```

Both JPG images and their matching XML metadata are required to run the analysis.
Use the 2015 training data only; do not mix in the test package. If extraction
adds another enclosing directory, move its actual `train/` folder to the location
shown above. Keep the downloaded ZIP out of Git. Retain the licence and attribution
information. The raw files are ignored by Git; only this placement guide is tracked.
The original research copy contains 91,758 image/XML pairs, whereas the official
description reports 91,759. Do not remove an arbitrary file to match the count.

The dataset can stay elsewhere on disk. Pass its containing directory with
`--data-dir` to `analyze_data.py`, `train.py` and `evaluate.py`.
Use a different directory for any future dataset and add its raw files to `.gitignore`.
