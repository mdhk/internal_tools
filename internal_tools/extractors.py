import copy
import torch
import numpy as np
import re
from itertools import chain
from collections.abc import Iterable
from numpy.typing import ArrayLike

from internal_tools.internal_utils import SaveOutput
from internal_tools.model_registry import (
    SUPPORTED_AUDIO_MODELS, 
    name_list,
    library_list,
    load_default_extractor_config
)

class AudioModelExtractor:
    def __init__(
        self, 
        model,
        activation_components: dict = None,
        component_transpose_dims: dict = {},
        component_transforms: dict = {}
    ):
        assert model.__class__.__name__ in name_list(SUPPORTED_AUDIO_MODELS), "Provided model should be in this list: \n\n" +\
            f"{library_list(SUPPORTED_AUDIO_MODELS)}\nInstead got: {type(model)}."

        self.model = model

        if activation_components == None:
            (
                self._component_descriptors, 
                self._component_transpose_dims,
                self._component_transforms
            ) = load_default_extractor_config(self.model)
        else:
            self._component_descriptors = copy.deepcopy(activation_components)
            self._component_transpose_dims = copy.deepcopy(component_transpose_dims)
            for comp_name, desc in self._component_descriptors.items():
                if 'conv' in desc and not comp_name in self._component_transpose_dims.keys():
                    self._component_transpose_dims[comp_name] = (1, -1)
            self._component_transforms = copy.deepcopy(component_transforms)

        assert all([k in self._component_descriptors.keys() for k in self._component_transpose_dims]), \
            "All keys in the component_transpose_dims dict should correspond to keys in the " +\
            "activation_components dict"
        assert all([
            k in self._component_descriptors.keys() 
            for k in chain(*[v[1] for v in self._component_transforms.values()])
            ]), \
            "All input components defined in the component_transforms dict should correspond to keys " +\
            "activation_components dict"

        self._activation_components = {
            comp: self._get_model_subcomp(model, self._component_descriptors[comp])
            for comp in self._component_descriptors.keys()
        }
        self._save_output = SaveOutput()
        
        for component_name, model_component in self._activation_components.items():
            model_component.register_forward_hook(self._save_output(component_name))

    def clear(self):
        self._save_output.clear()

    def get_activations(
        self, 
        between_times: list = None, 
        input_size: int = None, 
        samp_freq: int = 16000,
        to_cpu = True, 
        pooling: str | None = None
    ):
        """
        between_times: 
            list of tuples with start and end times of audio segments to extract activations for, in seconds;
            or list of lists with start and end times for each input in batch;
            if None, return all activations
        input_size: total size of the (padded) audio input(s), in samples;
            necessary when extracting activations between specified times, to compute input to activation ratio;
            should be an integer also for batch inputs, as all inputs in batch have the same size after padding
        samp_freq: sampling frequency of the audio input, in Hz
            should be 16000 unless the model is pre-trained with a different sampling rate
        to_cpu: whether to move the extracted activations to cpu
        pooling: pooling over time frames within each segment ('mean', 'middle' or None; no pooling if None)
        """
        
        assert len(self._save_output.outputs) != 0, "No extracted activations; run model forward pass first"

        comp_activations = self._create_activations_dict()

        for comp in comp_activations.keys():
            # apply any defined transforms
            if comp in self._component_transforms.keys():
                func, input_component_names, _ = self._component_transforms[comp]
                input_comp_activations = tuple(
                    self._get_component_activations(comp_name)
                    for comp_name in input_component_names
                )
                comp_activations[comp] = func(*input_comp_activations)
            else:
                comp_activations[comp] = self._get_component_activations(comp)
        
        # if time segments are not specified, return all activations (optionally pool and/or move to cpu)
        if between_times == None:
            comp_activations = {
                comp: self._to_cpu(
                    self._pooling(
                        comp_activations[comp],
                        pooling=pooling
                        ),
                    to_cpu=to_cpu
                    )
                for comp in comp_activations.keys()
            }

            return comp_activations

        # if time segments are specified, collect activations from specified time segments
        elif type(between_times) == list and isinstance(between_times[0], Iterable):
            assert input_size != None, \
            "Must provide input_size (length of audio input(s) in samples) when extracting activations between specified times"
            
            if type(between_times[0]) == list:
                assert len(between_times) == self._save_output.outputs[list(self._save_output.outputs.keys())[0]].shape[0], \
                f"len(between_times) does not match batch_size: " +\
                f"{len(between_times)} != {self._save_output.outputs[list(self._save_output.outputs.keys())[0]].shape[0]}"
                is_batch = True

            else:
                assert all([len(tup) == 2 and tup[0] < tup[1] for tup in between_times]), \
                "between_times should be a list of (start_time, end_time) pairs with start_time < end_time for each pair; " \
                "or a list of lists when extracting activations for multiple inputs"
                is_batch = False
            
            for comp in comp_activations:
                segment_frame_indices = self._get_frame_indices(
                    N_frames=comp_activations[comp].shape[1],
                    input_size=input_size,
                    time_segments=between_times,
                    samp_freq=samp_freq,
                    batch=is_batch
                )

                comp_activations[comp] = [
                    [
                        self._to_cpu(
                            self._pooling(
                                comp_activations[comp][input_i, segment_frame_indices[input_i][segment_s], :], 
                                pooling=pooling
                            ),
                            to_cpu=to_cpu
                        )
                        for segment_s in range(len(segment_frame_indices[input_i]))
                    ] 
                    for input_i in range(len(segment_frame_indices))
                ]

            if is_batch:
                return comp_activations
            else:
                if to_cpu:
                    return {comp: np.array([a.numpy() for a in acts[0]], dtype=object) for comp, acts in comp_activations.items()}
                else:
                    return {comp: acts[0] for comp, acts in comp_activations.items()}

        else:
            raise AssertionError(
                "between_times should be a list of (start_time, end_time) pairs specifying the time segments to extract activations from"
            )
        

    def get_indexed_segment_activations(
        self, 
        segment_index: ArrayLike,
        segment_start_times: ArrayLike,
        segment_end_times: ArrayLike,
        input_size: int,
        to_cpu: bool = True,
        pooling: str | None = None
    ):
        """
        Return flattened array of segment activations along with corresponding indices specified by segment_index.
        segment_index:
            should be of shape (batch_size, N_segments) and uniquely identify each segment to 
            extract activations for.
        segment_start_times and segment_end_times:
            should also be of shape (batch_size, N_segments) and specify the start and end times 
            (in seconds) for each segment respectively.
        """
        assert len(segment_index) == len(segment_start_times) == len(segment_end_times), \
        "segment_index, segment_start_times, segment_end_times should be of the same length (= batch size) but are " +\
        f"{len(segment_index)}, {len(segment_start_times)}, {len(segment_end_times)}"
        
        between_times = [list(zip(segment_start_times[f], segment_end_times[f])) for f in range(len(segment_index))]
        activations = self.get_activations(between_times=between_times, input_size=input_size, to_cpu=to_cpu, pooling=pooling)

        # flatten activations and indices
        for comp in activations.keys():
            activations[comp] = [a for acts in activations[comp] for a in acts]
        actv_indices = [i for indices in segment_index for i in indices]
        
        return actv_indices, activations
    
    def _get_component_activations(
        self,
        comp_name
    ): 
        # reorder activation axes where needed such that hidden_dim is last
        # to ensure all shapes are (batch_size, N_frames, hidden_dim)
        if comp_name in self._component_transpose_dims:
            return self._save_output.outputs[comp_name].transpose(*self._component_transpose_dims[comp_name])
        else:
            return self._save_output.outputs[comp_name]

    def _get_model_subcomp(
        self, 
        subcomp: torch.nn.Module, 
        target_subcomp_str: str
    ) -> torch.nn.Module:
        """
        Return the model subcomponent given by target_subcomp_str if it exists, otherwise 
        throw an error. This follows the get_subcomponent function for torch.nn.Module, 
        except we also parse indices to layer stacks (e.g. "conv_layers[-1]").
        """
        if target_subcomp_str == '':
            return subcomp
        else:
            subcomp_atoms: list[str] = target_subcomp_str.split('.')
            
            for atom in subcomp_atoms:
                subcomp_list = re.findall(r'[^\[\]]+', atom)
                
                if not hasattr(subcomp, subcomp_list[0]):
                    raise AttributeError(
                        f"{subcomp._get_name()} has no attribute `{subcomp_list[0]}`"
                    )
                    
                if len(subcomp_list) == 2:
                    subcomp_str, subcomp_idx = subcomp_list[0], int(subcomp_list[1])
                    subcomp = getattr(subcomp, subcomp_str)[subcomp_idx]
                elif len(subcomp_list) == 1:
                    subcomp_str = subcomp_list[0]
                    subcomp = getattr(subcomp, subcomp_str)
                else:
                    raise NotImplementedError(f"Can't parse subcomp_str: {subcomp_str}")
        
                if not isinstance(subcomp, torch.nn.Module):
                    raise AttributeError(f"`{subcomp}` is not a torch.nn.Module")   
                    
            return subcomp

    def _get_frame_indices(
        self, 
        N_frames: int, 
        input_size: int, 
        time_segments: list, 
        samp_freq: int = 16000,
        batch = True
    ):
        input_activations_ratio = input_size // N_frames
        frame_rate = input_activations_ratio / samp_freq
        
        if batch:
            frame_indices = []
            for input_i in range(len(time_segments)):
                input_frame_indices = []
                for start_time, end_time in time_segments[input_i]:
                    input_frame_indices.append(list(range(int(np.floor(float(start_time) / frame_rate)),
                                                          int(np.ceil(float(end_time) / frame_rate))+1)))
                frame_indices.append(input_frame_indices)
        else:
            frame_indices = [[]]
            for start_time, end_time in time_segments:
                frame_indices[0].append(list(range(int(np.floor(float(start_time) / frame_rate)),
                                                          int(np.ceil(float(end_time) / frame_rate))+1)))
        return frame_indices

    def _pooling(
        self, 
        activations: torch.Tensor, 
        pooling: str | None
    ):
        assert pooling in ['mean', 'middle', None], "pooling must be one of ['mean', 'middle', None]"

        if pooling == None:
            return activations
        elif pooling == 'middle':
            middle_idx = activations.squeeze().shape[0]//2
            return activations[middle_idx, :]
        elif pooling == 'mean':
            return activations.squeeze().mean(axis=0)
    
    def _to_cpu(
        self,
        activations: torch.Tensor,
        to_cpu: bool
    ):
        if to_cpu:
            return activations.cpu()
        else:
            return activations
        
    def _create_activations_dict(
        self
    ):
        component_names = list(self._component_descriptors.keys())
        for transformed_component, (_, components_to_transform, keep_inputs) in self._component_transforms.items():
            comptt_idx = [component_names.index(ctt) for ctt in components_to_transform]
            component_names.insert(comptt_idx[-1], transformed_component)
            comptt_idx[-1] += 1
            if not keep_inputs:
                component_names = [
                    comp_name for c, comp_name in enumerate(component_names)
                    if not c in comptt_idx
                ]
        return {comp_name: [] for comp_name in component_names}