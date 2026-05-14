import torch
from transformers import (
    Wav2Vec2Model, 
    Wav2Vec2ForCTC,
    HubertModel,
    HubertForCTC,
    WavLMModel,
    WavLMForCTC
)
from spidr.models.spidr import SpidR
from types import ModuleType

from internal_tools.internal_utils import define_extractor_config, define_transform


SUPPORTED_AUDIO_MODELS = [
    Wav2Vec2Model, 
    Wav2Vec2ForCTC,
    HubertModel,
    HubertForCTC,
    WavLMModel,
    WavLMForCTC,
    SpidR
]
SUPPORTED_MODELS = SUPPORTED_AUDIO_MODELS

library_dict = {
    m: m.__module__.split('.')[0]
    for m in SUPPORTED_MODELS
}

def name_list(model_list: list) -> str:
    """
    Returns a string listing names for all models in model_list.
    """
    return f"[{', '.join([m.__class__.__name__ for m in model_list])}]"

def library_list(model_list: list, library: str | ModuleType = None) -> str:
    """
    Returns a string listing model names per library.
    If library is specified, list only models for that library;
    else list all models.
    """
    if library and type(library) != str:
        library = library.__name__

    libraries = [m.__module__.split('.')[0] for m in model_list]
    model_names = [m.__class__.__name__ for m in model_list]
    
    col1_w = max([len(l) for l in libraries])
    col2_w = max([len(mn) for mn in model_names])

    libstr = f"{'library':{col1_w}}\t{'model':{col2_w}}\n" +\
            f"{'='*col1_w}\t{'='*col2_w}\n"
    
    for m in range(len(model_list)):
        if not library or libraries[m] == library:
            libstr += f"{libraries[m]:{col1_w}}\t{model_names[m]:{col2_w}}\n"

    return libstr

def load_default_extractor_config(model):
    """
    Load default extractor configuration for the provided model.
    """
    if type(model) in SUPPORTED_AUDIO_MODELS:
        if library_dict[type(model)] == 'transformers' and 'ForCTC' in model.__class__.__name__:
            prefix = model.__class__.__name__.strip('ForCTC').lower()
            model_comps = model.getattr(prefix)
        else:
            prefix = ''
            model_comps = model

        # specify component_descriptors: 
        # submodule reference strings defining where to extract activations from
        if library_dict[type(model)] == 'transformers':
            component_descriptors = {
                'CNN': f'{prefix}feature_extractor.conv_layers[-1].activation',
                'feature_proj': f'{prefix}feature_projection.projection',
                'pos_embeds': f'{prefix}encoder.pos_conv_embed.activation' } \
            | {
                f'T{i+1}': f'{prefix}encoder.layers[{i}].final_layer_norm'
                for i in range(len(model_comps.encoder.layers))
            }
        elif library_dict[type(model)] == 'spidr':
            component_descriptors = {
                'CNN': f'feature_extractor.conv_layers[-1].conv',
                'feature_proj': f'feature_projection.projection',
                'pos_embeds': f'student.pos_conv_embed' } \
            | {
                f'T{i+1}': f'student.layers[{i}].final_layer_norm'
                for i in range(len(model_comps.student.layers))
            }
        else:
            raise NotImplementedError
        
        # specify component_transpose_dims: 
        # where & how activation axes should be reordered such that hidden_dim is last
        if library_dict[type(model)] == 'transformers':
            component_transpose_dims = {
                'CNN': (1, -1),
                'pos_embeds': (1, -1)
            }
        elif library_dict[type(model)] == 'spidr':
            component_transpose_dims = {
                'CNN': (1, -1)
            }
        else:
            raise NotImplementedError
        
        # specify component_transforms:
        # any transformations or operations to apply on the extracted components to
        # generate the desired output (e.g. components to merge)
        if library_dict[type(model)] in ['transformers', 'spidr']:
            component_transforms = {
                # should be a dict with items structured as follows
                # transformed_component: 
                #   define_tranform(function_to_apply, components_to_transform, keep_inputs)
                # for each item,
                # transformed_component 
                #   will be a key in the output activations dict
                # function_to_apply 
                #   will operate on the components labeled by components_to_transform
                # components_to_transform 
                #   should be keys in the component_descriptors dict
                # keep_inputs 
                #   should be a bool specifying whether to keep components_to_transform in
                #   the output activations dict (in addition to tranformed_component)
                'embeds': define_transform(torch.add, ['feature_proj', 'pos_embeds'], False)
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