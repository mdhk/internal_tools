import torch

from huggingface_hub import hf_hub_download

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    HubertModel,
    HubertForCTC,
    HubertConfig,
    WavLMModel,
    WavLMForCTC,
    WavLMConfig
)

from internal_tools.models import (
    load_CPC_model,
    CPCConfig,
    load_SpidR_model,
    SpidRConfig,
    load_MelHuBERT_model,
    load_BabyHuBERT_model
)
from internal_tools.preprocessors import AudioPreprocessor

def load_preprocessor(model_name):
    match model_name:
        case 'wav2vec2':
            return AudioPreprocessor.for_hf_model('facebook/wav2vec2-base')
        case 'wav2vec2-ctc': 
            return AudioPreprocessor.for_hf_model('facebook/wav2vec2-base-960h')
        case 'hubert': 
            return AudioPreprocessor.for_hf_model('facebook/hubert-base-ls960')
        case 'hubert-ctc': 
            return AudioPreprocessor.for_hf_model('facebook/hubert-large-ls960-ft')
        case 'babyhubert':
            return AudioPreprocessor.for_hf_model('MarvinLvn/BabyHuBERT')
        case 'wavlm': 
            return AudioPreprocessor.for_hf_model('microsoft/wavlm-base')
        case 'wavlm-ctc':
            return AudioPreprocessor.for_hf_model('patrickvonplaten/wavlm-libri-clean-100h-base')
        case 'spidr':
            return AudioPreprocessor.for_spidr_model()
        case 'cpc': 
            return AudioPreprocessor.for_cpc_model()
        case 'melhubert':
            return AudioPreprocessor.for_melhubert_model()

def load_model(model_name, trained=True):
    match model_name:
        case 'wav2vec2': 
            if trained:
                return Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
            else:
                return Wav2Vec2Model(Wav2Vec2Config())
        case 'wav2vec2-ctc': 
            if trained:
                return Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
            else:
                return Wav2Vec2ForCTC(Wav2Vec2Config())
        case 'hubert': 
            if trained:
                return HubertModel.from_pretrained('facebook/hubert-base-ls960')
            else:
                return HubertModel(HubertConfig())
        case 'hubert-ctc': 
            if trained:
                return HubertForCTC.from_pretrained('facebook/hubert-large-ls960-ft')
            else:
                return HubertForCTC(HubertConfig())
        case 'babyhubert':
            if trained:
                return load_BabyHuBERT_model(hf_hub_download(repo_id="MarvinLvn/BabyHuBERT", filename="BabyHuBERT.ckpt"))
            else:
                return load_BabyHuBERT_model()
        case 'wavlm': 
            if trained:
                return WavLMModel.from_pretrained('microsoft/wavlm-base')
            else:
                return WavLMModel(WavLMConfig())
        case 'wavlm-ctc':
            if trained:
                return WavLMForCTC.from_pretrained('patrickvonplaten/wavlm-libri-clean-100h-base')
            else:
                return WavLMForCTC(WavLMConfig())
        case 'spidr':
            if trained:
                return torch.hub.load("facebookresearch/spidr", "spidr_base")
            else:
                return load_SpidR_model(config=SpidRConfig())
        case 'cpc': 
            if trained:
                return load_CPC_model("models/cpc_checkpoint_106.pt")
            else:
                return load_CPC_model(config=CPCConfig())
        case 'melhubert':
            if trained:
                return load_MelHuBERT_model('models/melhubert_960_stage2_20ms.ckpt')
            else:
                return load_MelHuBERT_model(config='default_20ms')