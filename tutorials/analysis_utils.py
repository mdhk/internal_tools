import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression

def layer_depth_map(
    ordered_layer_names: list, 
    first_model_layer: str = 'embeds', 
    step_size_decimals = 2
):
    """
    Returns a dict with layer names as keys and depths as values.
    Will concatenate a range of negative and positive depths around first_model_layer 
    (i.e. if a range of feature encoder layers precede first_model_layer, they will be
    assigned negative depths).
    """
    if not first_model_layer in ordered_layer_names:
        warnings.warn(
            f"No layer with name '{first_model_layer}' in ordered_layer_names; " +\
            f"setting first_model_layer to '{ordered_layer_names[0]}' instead"
        )
        first_model_layer = ordered_layer_names[0]
    ordered_layer_names = list(ordered_layer_names)
    first_layer_idx = ordered_layer_names.index(first_model_layer)
    positive_layers = ordered_layer_names[first_layer_idx:]
    step_size = round(
            1/(max(1, len(positive_layers))), 
            step_size_decimals
        )
    positive_depth_range = np.arange(0, 1, step_size)
    if first_layer_idx > 0:
        negative_depth_range = np.arange(-(step_size*len(ordered_layer_names[:first_layer_idx])), 0, step_size)
    else:
        negative_depth_range = []
        
    layer_depth_map = dict(zip(
        ordered_layer_names, 
        [round(d, step_size_decimals) for d in np.concatenate([negative_depth_range, positive_depth_range]).tolist()]
    ))
    return layer_depth_map

def create_ABX_triplets(annotations, contrasts, condition='across-speaker'):
    """
    Create a dataset of phone ABX triplets based on the provided contrasts and annotations.
    If condition == 'within-speaker', items within triplets will be sampled from the same speaker.
    If condition == 'across-speaker', items within triplets will be sampled from different speakers.
    If condition == 'random', items will be a random sample from the annotations dataframe.
    """
    triplet_set = []
    for r, contrast_row in contrasts.iterrows():
        phoneA = contrast_row['A']
        phoneB = contrast_row['B']
        contrast_set = annotations[
            (annotations['phone_ipa'].isin([phoneA, phoneB]))
        ]
        for speaker in contrast_set['speaker_id'].unique():
            speaker_contrast_set = contrast_set[contrast_set['speaker_id'] == speaker]
            split_idx = len(speaker_contrast_set)//4
    
            if condition == 'within-speaker':
                ## triplets for testing sim(phoneA, phoneA) > sim(phoneA, phoneB)
                A1_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneA].index.values[:split_idx]
                X1_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneA].index.values[split_idx:]
                B1_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneB].index.values[:split_idx]
        
                ## triplets for testing sim(phoneB, phoneB) > sim(phoneA, phoneB)
                A2_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneB].index.values[:split_idx]
                X2_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneB].index.values[split_idx:]
                B2_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneA].index.values[:split_idx]
        
            elif condition == 'across-speaker':
                other_speakers = [sp for sp in contrast_set['speaker_id'].unique() if not sp == speaker]
    
                ## triplets for testing sim(phoneA, phoneA) > sim(phoneA, phoneB)
                A1_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneA].index.values[:split_idx]
                X1_idx = np.empty(A1_idx.shape).astype(int)
                B1_idx = np.empty(A1_idx.shape).astype(int)
                for i in range(len(A1_idx)):
                    speakerX, speakerB = np.random.choice(other_speakers, size=2, replace=False)
                    X1_idx[i] = contrast_set[(contrast_set['speaker_id'] == speakerX) & (contrast_set['phone_ipa'] == phoneA)].sample(1).index.item()
                    B1_idx[i] = contrast_set[(contrast_set['speaker_id'] == speakerB) & (contrast_set['phone_ipa'] == phoneB)].sample(1).index.item()
    
                ## triplets for testing sim(phoneB, phoneB) > sim(phoneA, phoneB)
                A2_idx = speaker_contrast_set[speaker_contrast_set['phone_ipa'] == phoneB].index.values[:split_idx]
                X2_idx = np.empty(A2_idx.shape).astype(int)
                B2_idx = np.empty(A2_idx.shape).astype(int)            
                for i in range(len(A2_idx)):
                    speakerX, speakerB = np.random.choice(other_speakers, size=2, replace=False)
                    X2_idx[i] = contrast_set[(contrast_set['speaker_id'] == speakerX) & (contrast_set['phone_ipa'] == phoneB)].sample(1).index.item()
                    B2_idx[i] = contrast_set[(contrast_set['speaker_id'] == speakerB) & (contrast_set['phone_ipa'] == phoneA)].sample(1).index.item()
    
            elif condition == 'random':
                sample_idx = np.random.choice(annotations.index.tolist(), size=split_idx*6, replace=False)
                A1_idx = sample_idx[:split_idx]
                B1_idx = sample_idx[split_idx:2*split_idx]
                X1_idx = sample_idx[2*split_idx:3*split_idx]
                A2_idx = sample_idx[3*split_idx:4*split_idx]
                B2_idx = sample_idx[4*split_idx:5*split_idx]
                X2_idx = sample_idx[5*split_idx:6*split_idx]
    
            else:
                raise AssertionError(f"{condition} is not a valid condition; choose from ['within-speaker', 'across-speaker', 'random']")
                    
            for A_idx, B_idx, X_idx in zip([*A1_idx, *A2_idx], [*B1_idx, *B2_idx], [*X1_idx, *X2_idx]):
                A_row = annotations.loc[A_idx]
                B_row = annotations.loc[B_idx]
                X_row = annotations.loc[X_idx]
                triplet_set.append(
                    (int(A_idx), int(B_idx), int(X_idx), 
                     A_row['phone_ipa'], B_row['phone_ipa'], X_row['phone_ipa'], 
                     A_row['speaker_id'], B_row['speaker_id'], X_row['speaker_id'],
                     A_row['speaker_sex'], B_row['speaker_sex'], X_row['speaker_sex'],
                     A_row['file_id'], B_row['file_id'], X_row['file_id'],
                     contrast_row['speech_sound_category'], contrast_row['contrast_feature'], condition)
                )
    
    triplet_set_df = pd.DataFrame(triplet_set, columns=[
            'A_idx', 'B_idx', 'X_idx', 'A_phone', 'B_phone', 'X_phone', 
            'A_speaker_id', 'B_speaker_id', 'X_speaker_id', 'A_speaker_sex', 'B_speaker_sex', 'X_speaker_sex',
            'A_file_id', 'B_file_id', 'X_file_id', 'speech_sound_category', 'contrast_feature', 'ABX_condition'
            ])
    
    return triplet_set_df

def get_ABX_scores(
        A_embs, 
        B_embs, 
        X_embs, 
        return_sims=True, 
        sim_func=torch.nn.CosineSimilarity(), 
        batch_size=50, 
        device=torch.device('cpu')
    ):
    """
    Return ABX accuracy scores (and optionally AX and BX similarities) for the provided A, B, and X embeddings.
    Accuracy is computed by checking if sim_func(A, X) > sim_func(B, X).
    """
    
    assert A_embs.shape == B_embs.shape == X_embs.shape, \
    "A, B and X embeddings must have identical shapes (N_samples, N_features); " +\
    f"instead got {A_embs.shape}, {B_embs.shape}, {X_embs.shape}"

    N_batches = int(np.ceil(len(A_embs)/batch_size))
    A_embs, B_embs, X_embs = [
        [torch.Tensor(a) for a in np.array_split(embs, N_batches)]
        for embs in [A_embs, B_embs, X_embs]
    ]

    AX_sims = []
    BX_sims = []
    accuracies = []

    for b in range(N_batches):
        A_embs[b].to(device), B_embs[b].to(device), X_embs[b].to(device)
        AX_sim = sim_func(A_embs[b], X_embs[b])
        BX_sim = sim_func(B_embs[b], X_embs[b])
        acc = AX_sim > BX_sim
        AX_sims.extend(AX_sim.detach().cpu().tolist())
        BX_sims.extend(BX_sim.detach().cpu().tolist())
        accuracies.extend(acc.detach().cpu().tolist())

    if return_sims:
        return accuracies, AX_sims, BX_sims
    else:
        return accuracies
    
def run_ABX_tests(
        embeddings: dict, 
        ABX_triplets: pd.DataFrame, 
        normalize=True, 
        device=torch.device('cpu'),
        first_model_layer='embeds'
    ):
    """
    Compute ABX scores across all provided ABX_triplets using the provided embeddings.

    All values of the embeddings dict should be arrays that can be indexed using the idx
    columns in the ABX_triplets dataframe, to retrieve the corresponding embedding.
    """
    layer_depths = layer_depth_map(embeddings.keys(), first_model_layer=first_model_layer)
    ABX_results = []
    
    for layer in tqdm(embeddings):
        if normalize:
            layer_embs = zscore(embeddings[layer])
        else:
            layer_embs = embeddings[layer]
            
        A_embs = layer_embs[ABX_triplets['A_idx'].values]
        B_embs = layer_embs[ABX_triplets['B_idx'].values]
        X_embs = layer_embs[ABX_triplets['X_idx'].values]
        
        accuracies, AX_sims, BX_sims = get_ABX_scores(A_embs, B_embs, X_embs, batch_size=len(A_embs), device=device)
        for a in range(len(accuracies)):
            ABX_results.append((
                layer, layer_depths[layer], *[ABX_triplets.loc[a][col] for col in ABX_triplets.columns],
                AX_sims[a], BX_sims[a], accuracies[a]
            ))
    ABX_df = pd.DataFrame(ABX_results, columns=['layer', 'depth', *ABX_triplets.columns, 'AX_sim', 'BX_sim', 'accuracy'])

    return ABX_df

def generate_fold_splits(annotations: pd.DataFrame, Nfolds: int = 5):
    # select one male and one female test speaker per fold
    test_speakers = {
        f: [
            annotations[annotations['speaker_sex'] == 'F']['speaker_id'].unique()[f].item(),
            annotations[annotations['speaker_sex'] == 'M']['speaker_id'].unique()[f].item()
        ] for f in range(Nfolds)
    }
    # split train & test data per fold
    fold_splits = {
        f: {
            'train': annotations[~annotations['speaker_id'].isin(test_speakers[f])].index.values.tolist(),
            'test': annotations[annotations['speaker_id'].isin(test_speakers[f])].index.values.tolist()
        } for f in range(Nfolds)
    }
    return fold_splits

def run_phone_probe(embeddings, annotations, fold_splits, normalize=True, randomize_labels=False, max_iter=1000,
                    include_columns=['phone_ipa', 'phone_position', 'phone_broad_category', 'phone_fine_category', 'speaker_id', 'speaker_sex'],
                    first_model_layer='embeds'):
    layer_depths = layer_depth_map(embeddings.keys(), first_model_layer=first_model_layer)
    probe_results = []

    if randomize_labels:
        annotations = annotations.rename(columns={col: f'RANDOMIZED_{col}' for col in include_columns})
        include_columns = [f'RANDOMIZED_{col}' for col in include_columns]

    for f in range(len(fold_splits)):
        train_idx = fold_splits[f]['train']
        test_idx = fold_splits[f]['test']

        test_annotations = annotations.loc[test_idx].reset_index(drop=True)
        
        if randomize_labels:
            train_labels = np.random.permutation(annotations.loc[train_idx]['RANDOMIZED_phone_ipa'].values)
            test_annotations = test_annotations.loc[np.random.permutation(test_annotations.index)].reset_index(drop=True)
            test_labels = test_annotations['RANDOMIZED_phone_ipa'].values
        else:
            train_labels = annotations.loc[train_idx]['phone_ipa'].values
            test_labels = test_annotations['phone_ipa'].values
            
        for layer in tqdm(embeddings, desc=f'fold {f}'):
            if normalize:
                layer_embs = zscore(embeddings[layer])
            else:
                layer_embs = embeddings[layer]
                
            train_embs = layer_embs[train_idx]
            test_embs = layer_embs[test_idx]
    
            logreg_probe = LogisticRegression(max_iter=max_iter)
            logreg_probe.fit(train_embs, train_labels)
    
            predictions = logreg_probe.predict(test_embs)
            accuracies = predictions == test_labels
            
            for p in range(len(predictions)):
                probe_results.append((
                    f, layer, layer_depths[layer], *[test_annotations.loc[p][col] for col in include_columns],
                    predictions[p], 'accuracy', accuracies[p]
                ))
                
    probe_results = pd.DataFrame(probe_results, columns=[
    'fold', 'layer', 'depth', *include_columns,
    'prediction', 'score_name', 'score'
    ])
    
    return probe_results