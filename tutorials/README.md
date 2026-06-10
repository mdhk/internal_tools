# `internal_tools` tutorials

This directory contains some example scripts and notebooks which demonstrate the use of `internal_tools`. 

## Colab notebooks
The .ipynb notebooks in this repository can be run on Google Colab:

0. [Sampling phone occurrences from LibriSpeech](https://colab.research.google.com/github/mdhk/internal_tools/blob/main/tutorials/0_sampling_librispeech_phones.ipynb)
1. [Extracting model activations for annotated audio recordings](https://colab.research.google.com/github/mdhk/internal_tools/blob/main/tutorials/1_extracting_model_activations.ipynb)
2. [Analyses of model-internal activations](https://colab.research.google.com/github/mdhk/internal_tools/blob/main/tutorials/2_activation_analyses.ipynb)

## Tutorial files
Running these tutorials requires some local files, specifically in the `data/` and `models/` subdirectories:
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