from .config import CPCConfig
from .model import CPCModel, CPCAR, CPCEncoder

import torch
import warnings
from pathlib import Path

class CPC(CPCModel):
    """
    This is a wrapper for the CPCModel defined in the model.py from the original
    CPC repository, with small adjustments for removing labels and hidden states 
    from the forward pass output, and transforming batched input data before it 
    enters the CNN.
    """
    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR= AR

    def forward(self, batchData, layer=None):
        if batchData.shape[0] > 1 and len(batchData.shape) != 3:
            batchData = batchData.unsqueeze(1)
        cFeature = None
        if layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            encodedData = self.gEncoder(batchData, layer)
        else:
            encodedData = self.gEncoder(batchData, layer).permute(0, 2, 1)
            cFeature = self.gAR(encodedData)
        return cFeature

def load_CPC_model(
        pretrained_checkpoint_path: str | Path = None, 
        config: CPCConfig = None, 
        state_dict_key = 'gEncoder',
        device=torch.device('cpu')
        ):
    if not config:
        config = CPCConfig()
    else:
        assert type(config) == CPCConfig, f"config must be of type {CPCConfig}"
    cpc_encoder = CPCEncoder(config.hiddenEncoder, config.normMode)
    cpc_ar = CPCAR(config.hiddenEncoder, 
        config.hiddenGar,
        config.samplingType == "sequential", 
        config.nLevelsGRU, 
        config.arMode,
        config.cpc_mode == "reverse"
    )
    cpc_model = CPC(cpc_encoder, cpc_ar)
    if pretrained_checkpoint_path:
        state_dict = torch.load(pretrained_checkpoint_path, map_location=device)
        assert state_dict_key in state_dict.keys(), \
            f"No '{state_dict_key}' found in the loaded model state, " +\
            "specify state_dict_key for loading the intended parameters " +\
            f"(available keys: {list(state_dict.keys())})"
        e = cpc_model.load_state_dict(state_dict[state_dict_key], strict=False)
        if e.missing_keys or e.unexpected_keys:
            warnings.warn(f"Loaded model state has incompatible keys: \n{e}")
    return cpc_model