import torch
import warnings
import numpy as np
from numpy.typing import ArrayLike
from librosa import resample
from transformers import AutoFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from torch.nn import functional as F

class AudioPreprocessor:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated for a specific model using the " +\
            f"{self.__class__.__name__}.for_hf_model(model_name_or_path) or {self.__class__.__name__}.for_spidr_model() " +\
            "methods"
        )
    
    @classmethod
    def for_hf_model(cls, model_name_or_path, *args, **kwargs):
        try: 
            preprocessor = AutoFeatureExtractor.from_pretrained(model_name_or_path, *args, **kwargs)
        except OSError as e:
            preprocessor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", *args, **kwargs)
            config_dict = preprocessor.to_dict()
            warnings.warn(
                f'No preprocessor_config.json file found for {model_name_or_path}, ' +\
                f'using default {config_dict["feature_extractor_type"]} instead. Settings: \n' +\
                config_dict
            )
        cls._check_padding_side(preprocessor)
        return preprocessor

    @classmethod
    def for_spidr_model(cls, *args, **kwargs):
        preprocessor = SpidRPreprocessor(*args, **kwargs)
        cls._check_padding_side(preprocessor)
        return preprocessor
    
    def _check_padding_side(preprocessor):
        if preprocessor.padding_side != 'right':
            warnings.warn(
            f"Preprocessor padding_side is configured to '{preprocessor.padding_side}', " +\
            "AudioModelExtractor functionality currently assumes padding_side == 'right'."
            )
    
class SpidRPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        padding_side: str = 'right',
        return_attention_mask: bool = False,
        do_normalize: bool = True
    ):
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.padding_side = padding_side
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        if self.return_attention_mask:
            raise NotImplementedError
        
    def __call__(
            self,
            audio_signals: ArrayLike,
            sampling_rate: int,
            padding: bool = False,
            return_tensors: str = None,
            padding_side: str = "right"
    ):
        if (return_tensors and return_tensors != "pt"):
            raise NotImplementedError

        # ensure input is contained within a list
        if isinstance(audio_signals[0], float):
            audio_signals = [audio_signals]

        # resample if needed
        if sampling_rate != self.sampling_rate:
            warnings.warn(
                f"Specified sampling_rate does not match preprocessor sampling rate ({sampling_rate} != {self.sampling_rate}), " +\
                "resampling audio signals to {self.sampling_rate} Hz to match preprocessor configuration."
                )
            audio_signals = [
                resample(s, orig_sr=sampling_rate, target_sr=self.sampling_rate)
                for s in audio_signals
            ]
        # pad input signals to be the same length
        if padding:
            audio_signals = self._pad_to_max_length(audio_signals, padding_side=padding_side)
        # convert inputs to torch tensors
        if return_tensors == "pt":
            try:
                audio_signals = self._to_torch(audio_signals)
            except ValueError as e:
                raise ValueError(
                    f"Unable to convert audio_signals to tensor: {e}. \n" +\
                    f"Use padding=True to pad all signals to the same length " +\
                    "or set return_tensors=None to return array of signals with different lengths."
                )
        # return numpy array if not torch tensors
        elif return_tensors == None:
            audio_signals = np.array(audio_signals, dtype=object)
        # normalize signals
        if self.do_normalize:
            # if processing a batch of audio signals with equal length, or a single audio signal
            try:
                audio_signals = self._normalize(audio_signals)
            # if processing a batch of unpadded audio signals (signals do not have equal length)
            except ValueError:
                audio_signals = np.array([self._normalize(s).numpy() for s in audio_signals], dtype=object)

        return BatchFeature({'input_values': audio_signals})

    def __repr__(self):
        return f'{self.__class__.__name__} ' +\
            str(self.__dict__).replace('{', '{\n ').replace(',', ',\n').replace('}', '\n}')

    def _pad_to_max_length(
        self,
        audio_signals,
        padding_side: str = 'right'
    ):
        max_length = max([len(s) for s in audio_signals])
        if padding_side == 'right':
            padded_signals = np.vstack([
                np.concat([np.array(s), np.zeros(max_length-len(s))], axis=-1)
                for s in audio_signals
            ])
        elif padding_side == 'left':
            padded_signals = np.vstack([
                np.concat([np.zeros(max_length-len(s)), np.array(s)], axis=-1)
                for s in audio_signals
            ])
        else:
            raise ValueError(
                f"Invalid padding_side: {padding_side}, must be one of: ['left', 'right']."
            )
        return padded_signals
    
    def _to_torch(
        self, 
        audio_signals: ArrayLike
    ):
        return torch.as_tensor(audio_signals, dtype=torch.float32)

    def _normalize(
        self, 
        audio: torch.Tensor | np.ndarray
    ):
        normalized_audio = F.layer_norm(torch.Tensor(audio), audio.shape)
        if isinstance(audio, torch.Tensor):
            return normalized_audio
        else:
            return normalized_audio.numpy()