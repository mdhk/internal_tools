"""
Example usage: python extract_embeddings.py --model_name='wav2vec2' --untrained
(This will extract embeddings from a randomly initialized wav2vec2 architecture)
"""

import torch
import gc
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

from internal_tools.extractors import AudioModelExtractor
from internal_tools.dataset_utils import get_annotated_audio_loader

from model_loading_utils import load_model, load_preprocessor

# reference names for a few example models to load for demonstration purposes
# (see model_loading_utils for the details of each model)
model_names = [
    'wav2vec2', 'wav2vec2-ctc', 'wavlm', 'wavlm-ctc', 'hubert', 'hubert-ctc', 
    'babyhubert', 'melhubert', 'spidr', 'cpc'
]

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
        help="whether to use a randomly initialized architecture instead of a trained model"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="data/libri_phone_sample/libri_phone_sample.csv",
        help="path to the file with the segment annotations to extract embeddings for"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/libri_phone_sample/audio",
        help="path to the directory with audio files accompanying the segment annotations"
    )
    parser.add_argument(
        "--emb_pooling",
        type=str,
        default='mean',
        choices=['mean', 'middle'],
        help="how to pool across time frames in embedding extraction"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='embeddings/',
        help="path to the directory in which to store extracted embeddings"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_init = 'untrained' if args.untrained else 'trained'
    print(f'Device: {device}')

    assert Path(args.annotations_file).exists(), f"File does not exist: {args.annotations_file}"
    assert Path(args.audio_dir).exists(), f"Directory does not exist: {args.audio_dir}"

    annotations = pd.read_csv(args.annotations_file)
    dl = get_annotated_audio_loader(annotations, Path(args.audio_dir), batch_size=50)

    model = load_model(args.model_name, trained=not(args.untrained))
    model.eval()
    model.to(device)
    preprocessor = load_preprocessor(args.model_name)
    extr = AudioModelExtractor(model)

    actv_indices = []
    activations = defaultdict(list)

    print(f'Annotations file: {args.annotations_file}\nAudio data: {args.audio_dir}')
    print(f'Extracting {len(annotations)} {args.emb_pooling}-pooled embeddings from {args.model_name} ({model_init} model)..')
    for b, batch_data in tqdm(enumerate(dl), total=len(dl)):
        inputs = preprocessor(
            batch_data['audio_signal'],
            sampling_rate=batch_data['audio_sampling_rate'],
            padding=True,
            return_tensors="pt",
            padding_side="right"
        ).to(device)
        max_audio_size = max([len(s) for s in batch_data['audio_signal']])

        with torch.no_grad():
            if args.model_name == 'melhubert':
                outputs = model(inputs.input_values, inputs.attention_mask)
            else:
                outputs = model(inputs.input_values)
        
        extracted_actv_indices, extracted_acts = extr.get_indexed_segment_activations(
            segment_index=batch_data['annotation_indices'], 
            segment_start_times=batch_data['annotation_start_times'],
            segment_end_times=batch_data['annotation_end_times'],
            input_size=max_audio_size, 
            pooling=args.emb_pooling
        )
        extr.clear()
        del inputs, outputs
        
        gc.collect()
        torch.cuda.empty_cache()

        actv_indices.extend(extracted_actv_indices)
        for layer in extracted_acts:
            activations[layer].extend(extracted_acts[layer])

    for layer in activations.keys():
        activations[layer] = np.stack([activations[layer][i] for i in np.argsort(actv_indices)])

    print('Done!')

    output_dir = Path(args.save_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(args.annotations_file).stem.replace('_','-')}_{args.emb_pooling}-embeddings_{args.model_name}-{model_init}.pkl"

    pickle.dump(activations, open(output_path, "wb"))
    print(f'Saved embeddings to {output_path}')