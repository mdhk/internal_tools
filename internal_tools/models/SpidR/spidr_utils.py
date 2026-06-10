from .config import SpidRConfig
from .model import SpidR

import torch
import warnings
from pathlib import Path

def load_SpidR_model(pretrained_checkpoint_path: str | Path = None, 
        config: SpidRConfig = None, 
        device=torch.device('cpu')
        ):
    if not config:
        config = SpidRConfig()
    else:
        assert type(config) == SpidRConfig, f"config must be of type {SpidRConfig}"
    
    spidr_model = SpidR(SpidRConfig())
    
    if pretrained_checkpoint_path:
        state_dict = torch.load(pretrained_checkpoint_path, map_location=device)
        e = spidr_model.load_state_dict(state_dict, strict=False)
        if e.missing_keys or e.unexpected_keys:
            warnings.warn(f"Loaded model state has incompatible keys: \n{e}")

    return spidr_model