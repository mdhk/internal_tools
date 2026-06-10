import torch
from transformers import (
    Wav2Vec2ForCTC,
    HubertModel,
    HubertForCTC,
    WavLMModel,
    WavLMForCTC
)
from transformers import Wav2Vec2Model as hfWav2Vec2Model
from torchaudio.models.wav2vec2.model import Wav2Vec2Model as taWav2Vec2Model
from internal_tools.models import CPC, SpidR, MelHuBERT
from types import ModuleType

from internal_tools.internal_utils import define_extractor_config, define_transform

SUPPORTED_AUDIO_MODELS = [
    taWav2Vec2Model,
    hfWav2Vec2Model, 
    Wav2Vec2ForCTC,
    HubertModel,
    HubertForCTC,
    WavLMModel,
    WavLMForCTC,
    SpidR,
    CPC,
    MelHuBERT
]
SUPPORTED_MODELS = SUPPORTED_AUDIO_MODELS    

def name_list(model_list: list) -> str:
    """
    Returns a string listing names for all models in model_list.
    """
    return f"[{', '.join([m.__name__ for m in model_list])}]"

def library_list(model_list: list, library: str | ModuleType = None) -> str:
    """
    Returns a string listing model names per library.
    If library is specified, list only models for that library;
    else list all models.
    """
    if library and type(library) != str:
        library = library.__name__

    libraries = [model_class.__module__.split('.')[0] for model_class in model_list]
    model_names = [model_class.__name__.split('.')[-1] for model_class in model_list]
    
    col1_w = max([len(l) for l in libraries])
    col2_w = max([len(mn) for mn in model_names])

    libstr = f"{'library':{col1_w}}\t{'model':{col2_w}}\n" +\
            f"{'='*col1_w}\t{'='*col2_w}\n"
    
    for m in range(len(model_list)):
        if not library or libraries[m] == library:
            libstr += f"{libraries[m]:{col1_w}}\t{model_names[m]:{col2_w}}\n"

    return libstr

def library(model):
    return f"{model.__module__.split('.')[0]}"

def name(model):
    return f"{model.__class__.__name__.split('.')[-1]}"

def load_default_extractor_config(model):
    """
    Load default extractor configuration for the provided model.
    """
    if name(model) in name_list(SUPPORTED_AUDIO_MODELS):
        if library(model) == 'transformers' and 'ForCTC' in name(model):
            prefix = name(model).strip('ForCTC').lower()
            model_comps = getattr(model, prefix)
            prefix = f"{prefix}."
        else:
            prefix = ''
            model_comps = model

        # specify component_descriptors: 
        # submodule reference strings defining where to extract activations from
        if library(model) == 'transformers':
            component_descriptors = {
                'CNN': f'{prefix}feature_extractor.conv_layers[-1].activation',
                'feature_proj': f'{prefix}feature_projection.projection',
                'pos_embeds': f'{prefix}encoder.pos_conv_embed.activation' } \
            | {
                f'T{i+1}': f'{prefix}encoder.layers[{i}].final_layer_norm'
                for i in range(len(model_comps.encoder.layers))
            }
        elif library(model) == 'torchaudio':
            component_descriptors = {
                'CNN': f'feature_extractor.conv_layers[-1].conv',
                'feature_proj': f'encoder.feature_projection.projection',
                'pos_embeds': f'encoder.transformer.pos_conv_embed' } \
            | {
                f'T{i+1}': f'encoder.transformer.layers[{i}].final_layer_norm'
                for i in range(len(model_comps.encoder.transformer.layers))
            }
        elif name(model) == 'SpidR':
            component_descriptors = {
                'CNN': f'feature_extractor.conv_layers[-1].conv',
                'feature_proj': f'feature_projection.projection',
                'pos_embeds': f'student.pos_conv_embed' } \
            | {
                f'T{i+1}': f'student.layers[{i}].final_layer_norm'
                for i in range(len(model_comps.student.layers))
            }
        elif name(model) == 'CPC':
            component_descriptors = {
                'CNN': 'gEncoder.batchNorm4',
                'LSTM': 'gAR.baseNet'
            }
        elif name(model) == 'MelHuBERT':
            component_descriptors = {
                'pre_embeds': 'pre_extract_proj',
                'pos_embeds': 'encoder.pos_conv' } \
            | {
                f'T{i+1}': f'encoder.layers[{i}].final_layer_norm'
                for i in range(len(model_comps.encoder.layers))
            }
        else:
            raise NotImplementedError
        
        # specify component_transpose_dims: 
        # where & how activation axes should be reordered 
        # such that hidden_dim is last and batch size is first
        if library(model) == 'transformers':
            component_transpose_dims = {
                'CNN': (1, -1),
                'pos_embeds': (1, -1)
            }
        elif library(model) == 'torchaudio' or name(model) in ['SpidR', 'CPC']:
            component_transpose_dims = {
                'CNN': (1, -1)
            }
        elif name(model) == 'MelHuBERT':
            component_transpose_dims = {
                'pos_embeds': (1, -1) } \
            | {
                f'T{i+1}': (0, 1)
                for i in range(len(model_comps.encoder.layers))
            }
        else:
            raise NotImplementedError
        
        # specify component_transforms:
        # any transformations or operations to apply on the extracted components to
        # generate the desired output (e.g. components to merge)
        #
        # `component_transforms` should be a dict with items structured as follows
        #        transformed_component: 
        #            define_tranform(function_to_apply, components_to_transform, keep_inputs)
        #
        #        where for each item,
        #        transformed_component 
        #            will be a key in the output activations dict
        #        function_to_apply 
        #             will operate on the components labeled by components_to_transform
        #        components_to_transform 
        #            should be keys in the component_descriptors dict
        #        keep_inputs 
        #            should be a bool specifying whether to keep components_to_transform in
        #            the output activations dict (in addition to tranformed_component)
        if library(model) in ['transformers', 'torchaudio'] or name(model) == 'SpidR':
            component_transforms = {
                'embeds': define_transform(torch.add, ['feature_proj', 'pos_embeds'], False)
            }
        elif name(model) == 'CPC':
            component_transforms = {}
        elif name(model) == 'MelHuBERT':
            component_transforms = {
                'embeds': define_transform(torch.add, ['pre_embeds', 'pos_embeds'], False)
            }
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    extractor_config = define_extractor_config(
        component_descriptors, 
        component_transpose_dims,
        component_transforms
        )
    
    return extractor_config