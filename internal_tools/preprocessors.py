import torch
import warnings
import numpy as np
import torchaudio
import copy
from numpy.typing import ArrayLike
from librosa import resample
from transformers import AutoFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from internal_tools.models.MelHuBERT.melhubert_utils import LIBRI_960_MEAN_STD

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
                str(config_dict)
            )
        cls._check_padding_side(preprocessor)
        return preprocessor

    @classmethod
    def for_spidr_model(cls, *args, **kwargs):
        preprocessor = WaveformPreprocessor(*args, **kwargs)
        cls._check_padding_side(preprocessor)
        return preprocessor
    
    @classmethod
    def for_cpc_model(cls, *args, **kwargs):
        if not 'do_normalize' in kwargs.keys():
            kwargs = {'do_normalize': False}
        preprocessor = WaveformPreprocessor(*args, **kwargs)
        cls._check_padding_side(preprocessor)
        return preprocessor
    
    @classmethod
    def for_melhubert_model(cls, *args, **kwargs):
        preprocessor = MelPreprocessor(*args, **kwargs)
        cls._check_padding_side(preprocessor)
        return preprocessor
    
    def _check_padding_side(preprocessor):
        if preprocessor.padding_side != 'right':
            warnings.warn(
            f"Preprocessor padding_side is configured to '{preprocessor.padding_side}', " +\
            "AudioModelExtractor functionality currently assumes padding_side == 'right'."
            )
    
class WaveformPreprocessor:
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
        
class MelPreprocessor:
    norm_mean, norm_std = LIBRI_960_MEAN_STD

    def __init__(
        self,
        sampling_rate: int = 16000,
        norm_mean: ArrayLike = norm_mean,
        norm_std: ArrayLike = norm_std,
        frame_period: int = 20,
        padding_value: float = 0.0,
        padding_side: str = 'right',
        return_attention_mask: bool = True,
        do_normalize: bool = True
    ):
        self.sampling_rate = sampling_rate
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.frame_period = frame_period
        self.padding_value = padding_value
        self.padding_side = padding_side
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        
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

        # Mel preprocessing & padding adapted from
        # https://github.com/nervjack2/MelHuBERT/blob/main/extract_feature.py
        mel_feats = [
            self._extract_fbank(signal, self.norm_mean, self.norm_std, fp=self.frame_period)
            for signal in audio_signals
        ]
        if padding:
            mel_lengths = [len(mel) for mel in mel_feats]
            mel_feats = pad_sequence(
                mel_feats, 
                batch_first=True, 
                padding_value=self.padding_value,
                padding_side=padding_side
            )
            # Prepare padding mask
            pad_mask = torch.ones(mel_feats.shape[:-1])
            # Zero vectors for padding dimension
            for idx in range(mel_feats.shape[0]):
                pad_mask[idx, mel_lengths[idx]:] = 0
        else:
            mel_feats = np.array([mf.detach().cpu().numpy() for mf in mel_feats], dtype=object)
            pad_mask = np.array([np.zeros(mf.shape) for mf in mel_feats], dtype=object)

        # to torch tensors
        if return_tensors == "pt":
            try:
                mel_feats = self._to_torch(mel_feats)
            except TypeError as e:
                raise TypeError(
                    f"Unable to convert audio_signals to tensor: {e}. \n" +\
                    f"Use padding=True to pad all signals to the same length " +\
                    "or set return_tensors=None to return array of signals with different lengths."
                )
        # return numpy array if not torch tensors
        elif return_tensors == None:
            mel_feats = np.array(mel_feats, dtype=object)

        if self.return_attention_mask:
            preproc_features =  BatchFeature({'input_values': mel_feats, 'attention_mask': pad_mask})
        else:
            preproc_features =  BatchFeature({'input_values': mel_feats})

        return preproc_features

    def __repr__(self):
        config_dict = copy.deepcopy(self.__dict__)
        del config_dict['norm_mean']
        del config_dict['norm_std']
        return f'{self.__class__.__name__} ' +\
            str(config_dict).replace('{', '{\n ').replace(',', ',\n').replace('}', '\n}')

    def _extract_fbank(self, waveform, mean, std, fp=20):
        """
        This function is adapted from the extract_fbank function in 
        https://github.com/nervjack2/MelHuBERT/blob/main/extract_feature.py
        """
        if waveform.dtype == object:
            waveform = waveform.astype(np.float32)
        waveform = torch.Tensor(waveform*(2**15)).unsqueeze(0)
        y = torchaudio.compliance.kaldi.fbank(
                            waveform,
                            num_mel_bins=40,
                            sample_frequency=16000,
                            window_type='hamming',
                            frame_length=25,
                            frame_shift=10)
        if self.do_normalize:
            # Normalize by the mean and std of Librispeech
            mean = mean.to(y.device, dtype=torch.float32)
            std = std.to(y.device, dtype=torch.float32)
            y = (y-mean)/std
        # Downsampling by twice 
        if fp == 20:
            odd_y = y[::2,:]
            even_y = y[1::2,:]
            if odd_y.shape[0] != even_y.shape[0]:
                even_y = torch.cat((even_y, torch.zeros(1,even_y.shape[1]).to(y.device)), dim=0)
            y = torch.cat((odd_y, even_y), dim=1)
        return y
    
    def _to_torch(
        self, 
        audio_signals: ArrayLike
    ):
        return torch.as_tensor(audio_signals, dtype=torch.float32)