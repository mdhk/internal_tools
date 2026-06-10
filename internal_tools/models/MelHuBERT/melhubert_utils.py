import numpy as np
import torch
import yaml
import warnings
from pathlib import Path

from .model import MelHuBERTModel, MelHuBERTConfig

def load_mean_std(
        mean_std_npy_path: Path | str = Path(__file__).parent / 'libri-960-mean-std.npy'
):
    """
    This is a modified version of the load_mean_std function from
    https://github.com/nervjack2/MelHuBERT/blob/main/extract_feature.py
    """
    assert Path(mean_std_npy_path).exists(), f"{mean_std_npy_path} does not exist; " +\
        "pre-computed values are provided at " +\
        "https://github.com/nervjack2/MelHuBERT/tree/main/example"
    mean_std = np.load(mean_std_npy_path)
    mean = torch.Tensor(mean_std[0].reshape(-1))
    std = torch.Tensor(mean_std[1].reshape(-1))
    return mean, std

def load_config_yaml(config_path: Path | str = None, frame_period: int = 20):
    if not config_path:
        config_path = Path(__file__).parent / f'config_model_{frame_period}ms.yaml'
    else:
        config_path = Path(config_path)
    assert config_path.exists(), f"No config file found at {config_path}"
    config = yaml.safe_load(config_path.read_text())['melhubert']
    return config

class MelHuBERT(MelHuBERTModel):
    """
    This is a wrapper for the MelHuBERTModel defined in the model.py from the original
    MelHuBERT repository, with small adjustments for the default arguments and output of the 
    forward pass.
    """
    def forward(
        self, 
        feat, 
        pad_mask, 
        cluster_label=None, 
        no_pred=True,
        mask=False, 
        get_hidden=False, 
    ):
        """
        Forward function
        Input:
            feat (FloatTensor): B x T_wave x D
            pad_mask (BoolTensor): B x T_wave
        """
        # Masking before projection 
        if mask and self.model_config.mask_before_proj:
            input_feat, mask_indices = self.apply_mask(feat, ~pad_mask.bool())
        else:
            input_feat = feat
            mask_indices = torch.full(pad_mask.shape, False)

        pre_feat = input_feat
        if self.pre_extract_proj != None:
            pre_feat = self.pre_extract_proj(input_feat)

        # Masking after projection 
        if mask and not self.model_config.mask_before_proj:
            x, mask_indices = self.apply_mask(pre_feat, ~pad_mask.bool())
        else:
            x = pre_feat
            mask_indices = mask_indices
        
        layer_hiddens = []
        hidden, layer_hiddens = self.encoder(
            x, ~pad_mask.bool(), get_hidden=get_hidden
        )
        
        if no_pred:
            return hidden, None, None, None, None, layer_hiddens, pre_feat

        assert cluster_label != None

        if not self.model_config.skip_masked:
            masked_indices = torch.logical_and(pad_mask.bool(), mask_indices)
            logit_m = self.final_proj(hidden[masked_indices])  # (num_masked, dim) -> (num_masked, num_cluster)
            label_m = cluster_label[masked_indices]
        else:
            logit_m = None
            label_m = None

        if not self.model_config.skip_nomask:
            nomask_indices = torch.logical_and(pad_mask.bool(), ~mask_indices) 
            logit_u = self.final_proj(hidden[nomask_indices])  # (num_unmask, dim) -> (num_unmask, num_cluster)
            label_u = cluster_label[nomask_indices]
        else:
            logit_u = None
            label_u = None
        
        return hidden, logit_m, logit_u, label_m, label_u, layer_hiddens, pre_feat, mask_indices
    
LIBRI_960_MEAN_STD = load_mean_std()
MELHUBERT_10MS_CONFIG = load_config_yaml(frame_period=10)
MELHUBERT_20MS_CONFIG = load_config_yaml(frame_period=20)

def load_MelHuBERT_model(
        pretrained_checkpoint_path: str | Path = None, 
        config: str | MelHuBERTConfig = None, 
        state_dict_key = 'model',
        device=torch.device('cpu')
        ):
    if not config or config == 'default_20ms':
        config = MelHuBERTConfig(MELHUBERT_20MS_CONFIG)
    elif config == 'default_10ms':
        config = MelHuBERTConfig(MELHUBERT_10MS_CONFIG)
    else:
        assert type(config) == MelHuBERTConfig, f"config must be of type {MelHuBERTConfig}"
    
    melhubert_model = MelHuBERT(config)

    if pretrained_checkpoint_path:
        state_dict = torch.load(pretrained_checkpoint_path, weights_only=False, map_location=device)
        assert state_dict_key in state_dict.keys(), \
            f"No '{state_dict_key}' found in the loaded model state, " +\
            "specify state_dict_key for loading the intended parameters " +\
            f"(available keys: {list(state_dict.keys())})"
        e = melhubert_model.load_state_dict(state_dict[state_dict_key], strict=False)
        if e.missing_keys or e.unexpected_keys:
            warnings.warn(f"Loaded model state has incompatible keys: \n{e}")

    return melhubert_model