# `internal_tools` tutorials

This directory contains some example scripts and notebooks which demonstrate the use of `internal_tools`. Running these tutorials requires some local files to be downloaded into this directory first, specifically into the `data/` and `models/` subdirectories:
```
.
├── data
│   ├── analysis_files/
│   │   ├── phone_ABX/
│   │   │   [data needed for running ABX tests]
│   │   └── phone_probe/
│   │       [data needed for running probes]
│   └── libri_phone_sample/
|       [data needed for activation extraction & analysis]
├── models
|   [local checkpoints for a CPC and a MelHuBERT model]
├── embeddings
|   [optional; pre-computed embeddings for example models]
└── results
    [optional; pre-computed analysis results for example models]
```
These files can be downloaded from [here](https://drive.google.com/drive/folders/1gngzccKa_87UEc5afiwS0XAF_ddXrUtU?usp=drive_link), or by using the `download_tutorial_files.sh` script:
```
./download_tutorial_files.sh data models
```

## Example
Running these commands will extract layerwise embeddings from a Wav2Vec2 model and compute phone ABX and probing classifier scores:
```
python extract_embeddings.py --model_name='wav2vec2'
```
```
python run_ABX_tests.py --model_name='wav2vec2' --condition='across-speaker'
```
```
python run_probing_classifiers.py --model_name='wav2vec2'
```