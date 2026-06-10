"""
Example usage: python run_probing_classifiers.py --model_name='wav2vec2' --untrained
(This will compute probing classifier scores on embeddings from a randomly initialized wav2vec2 architecture)
"""

import torch
import pickle
import json

import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

from analysis_utils import generate_fold_splits, run_phone_probe

# reference names for a few example models to load for demonstration purposes, see model_loading_utils for the details of each model
model_names = ['wav2vec2', 'wav2vec2-ctc', 'hubert', 'hubert-ctc', 'babyhubert', 'wavlm', 'wavlm-ctc', 'spidr', 'cpc', 'melhubert']

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        choices=model_names,
        help="name of the model to analyse"
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="whether to use embeddings from a randomly initialized architecture instead of a trained model"
    )
    parser.add_argument(
        "--randomize_labels",
        action="store_true",
        help="whether to randomize the phone labels before training probing classifiers (for control analyses)"
    )
    parser.add_argument(
        "--fold_splits_file",
        type=str,
        default="data/analysis_files/phone_probe/fold_splits.json",
        help="path to the file defining train-test splits to use for each fold " +\
        "(a fold_splits file will be generated if it does not exist)"
    )
    parser.add_argument(
        "--embeddings_file",
        default="embeddings/libri-phone-sample_mean-embeddings_{MODELNAME}-{MODELINIT}.pkl",
        type=str,
        help="path (template) to the file with precompputed embeddings"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="data/libri_phone_sample/libri_phone_sample.csv",
        help="path to the file with phone annotations for the extracted embeddings"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='results/',
        help="path to the directory in which to store analysis results"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    model_init = 'untrained' if args.untrained else 'trained'
    label_setting = 'randomized' if args.randomize_labels else 'true'

    emb_filepath = Path(
        args.embeddings_file.replace('{MODELNAME}', args.model_name).replace('{MODELINIT}', model_init)
    )
    assert emb_filepath.exists(), f"Embeddings file does not exist: {args.embeddings_file} \n" +\
    "Use --embeddings_file to point to a .pkl file as precomputed by running extract_embeddings.py"

    annotations_filepath = Path(args.annotations_file)
    assert annotations_filepath.exists(), f"Annotations file does not exist: {args.annotations_file} \n" +\
    "Use --annotations_file to point to a .csv file with annotations for the provided embeddings"
    
    annotations = pd.read_csv(annotations_filepath)

    fold_splits_filepath = Path(args.fold_splits_file)
    if not fold_splits_filepath.exists():
        print(f"No fold splits file found at {fold_splits_filepath}")
        print("Generating new train-test splits across 5 folds...")
        fold_splits = generate_fold_splits(annotations, Nfolds=5)
        if not fold_splits_filepath.parent.exists():
            fold_splits_filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(fold_splits_filepath, 'w') as fp:
            json.dump(fold_splits, fp)
        print(f"Done! Saved new fold splits to {fold_splits_filepath}")
    
    embeddings = pickle.load(open(emb_filepath, 'rb'))
    fold_splits = {int(k): v for k, v in json.load(open(fold_splits_filepath, 'r')).items()}
    
    # layer to center in computing layer depths
    first_model_layer = 'embeds' if not args.model_name == 'cpc' else 'LSTM'

    print(f'Annotations: {annotations_filepath} [{label_setting} labels]')
    print(f'Embeddings: {emb_filepath}')
    print('Computing probe scores...')
    probe_results = run_phone_probe(
        embeddings, annotations, fold_splits, 
        randomize_labels=args.randomize_labels,
        first_model_layer=first_model_layer
    )
    print('Done!')

    output_dir = Path(args.save_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"probe-results_{label_setting}-labels_{args.model_name}-{model_init}.csv"

    probe_results.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')