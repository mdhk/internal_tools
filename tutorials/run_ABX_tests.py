"""
Example usage: python run_ABX_tests.py --model_name='wav2vec2' --condition='across-speaker' --untrained
(This will compute across-speaker ABX scores on embeddings from a randomly initialized wav2vec2 architecture)
"""

import torch
import pickle

import pandas as pd

from pathlib import Path
from argparse import ArgumentParser

from analysis_utils import create_ABX_triplets, run_ABX_tests

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
        "--condition",
        required=True,
        type=str,
        choices=['within-speaker', 'across-speaker', 'random'],
        help="condition for the ABX tests to run"
    )
    parser.add_argument(
        "--triplets_file",
        default="data/analysis_files/phone_ABX/ABX_triplets_{CONDITION}.csv",
        type=str,
        help="path (template) to the file containing the triplets to use for the ABX test " +\
        "(a triplets file will be generated if it does not exist)"
    )
    parser.add_argument(
        "--embeddings_file",
        default="embeddings/libri-phone-sample_mean-embeddings_{MODELNAME}-{MODELINIT}.pkl",
        type=str,
        help="path (template) to the file with precompputed embeddings"
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="whether to use embeddings from a randomly initialized architecture instead of a trained model"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="data/libri_phone_sample/libri_phone_sample.csv",
        help="path to the file with phone annotations for the extracted embeddings"
    )
    parser.add_argument(
        "--contrasts_file",
        type=str,
        default="data/analysis_files/phone_ABX/phone_contrasts.csv",
        help="path to the file with phone contrasts to generate ABX triplets for"
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

    triplet_filepath = Path(
        args.triplets_file.replace('{CONDITION}', args.condition)
    )
    if not triplet_filepath.exists():
        print(f"No triplets file found at {triplet_filepath}")
        msg = "Annotations and contrasts files need to be provided for generating new ABX triplets."
        assert Path(args.annotations_file).exists(), f"Annotations file does not exist: {args.annotations_file} \n{msg}"
        assert Path(args.contrasts_file).exists(), f"Contrasts file does not exist: {args.contrasts_file} \n{msg}"
        print(f"Generating new set of triplets...")
        annotations = pd.read_csv(args.annotations_file)
        contrasts = pd.read_csv(args.contrasts_file)
        triplet_set_df = create_ABX_triplets(annotations, contrasts, condition=args.condition)
        if not triplet_filepath.parent.exists():
            triplet_filepath.parent.mkdir(exist_ok=True, parents=True)
        triplet_set_df.to_csv(triplet_filepath)
        print(f"Done! Saved new set of triplets to {triplet_filepath}")

    model_init = 'untrained' if args.untrained else 'trained'
    emb_filepath = Path(
        args.embeddings_file.replace('{MODELNAME}', args.model_name).replace('{MODELINIT}', model_init)
    )
    assert emb_filepath.exists(), f"Embeddings file does not exist: {args.embeddings_file} \n" +\
    "Use --embeddings_file to point to a .pkl file precomputed by running extract_embeddings.py"

    ABX_triplets = pd.read_csv(triplet_filepath)
    embs = pickle.load(open(emb_filepath, 'rb'))

    # layer to center in computing layer depths
    first_model_layer = 'embeds' if not args.model_name == 'cpc' else 'LSTM'

    print(f'Triplets file: {triplet_filepath}')
    print(f'Embeddings: {emb_filepath}')
    print('Computing ABX scores...')
    ABX_results = run_ABX_tests(
        embs, 
        ABX_triplets, 
        device=device, 
        first_model_layer=first_model_layer
    )
    print('Done!')

    output_dir = Path(args.save_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ABX-results_{args.condition}_{args.model_name}-{model_init}.csv"

    ABX_results.to_csv(output_path, index=False)
    print(f'Saved results to {output_path}')