from pathlib import Path
import torch
from torchaudio.models import hubert_pretrain_base

def load_BabyHuBERT_model(
        pretrained_checkpoint_path: str | Path = None, 
        device=torch.device('cpu')
        ):
    """
    This function is adapted from the code provided in the BabyHuBERT model card at
    https://huggingface.co/MarvinLvn/BabyHuBERT
    """
    model = hubert_pretrain_base(num_classes=500)
    if pretrained_checkpoint_path:
        state_dict = torch.load(pretrained_checkpoint_path, map_location=device)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
        model.load_state_dict(state_dict)
    encoder = model.wav2vec2
    return encoder