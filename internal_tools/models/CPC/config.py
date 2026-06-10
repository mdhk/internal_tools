# This file adapts code from
# https://github.com/facebookresearch/CPC_audio/blob/main/cpc/cpc_default_config.py
# to define a default configuration for the CPC model.
# 
# Copyright (c) Facebook, Inc. and its affiliates.
#
# The source code linked above is licensed under the MIT license found in the
# LICENSE file in the CPC directory.

class CPCConfig:
    """
    This class adopts the default configuration for the CPC model as defined in
    https://github.com/facebookresearch/CPC_audio/blob/main/cpc/cpc_default_config.py

    hiddenEncoder (`int`, *optional*, defaults to 256):
        Hidden dimension of the encoder network.
    hiddenGar (`int`, *optional*, defaults to 256):
        Hidden dimension of the auto-regressive network.
    nPredicts (`int`, *optional*, defaults to 12):
        Number of steps to predict.
    negativeSamplingExt (`int`, *optional*, defaults to 128):
        Number of negative samples to take.
    learningRate (`float`, *optional*, defaults to 2e-4):
        Learning rate.
    schedulerStep (`int`, *optional*, defaults to -1):
        Step of the learning rate scheduler: at each step the learning rate is divided by 2. 
        Default: no scheduler.
    schedulerRamp (`int`, *optional*, defaults to None):
        Enable a warm up phase for the learning rate: adds a linear ramp of the given size.
    beta1 (`float`, *optional*, defaults to 0.9):
        Value of beta1 for the Adam optimizer.
    beta2 (`float`, *optional*, defaults to 0.999):
        Value of beta2 for the Adam optimizer.
    epsilon (`float`, *optional*, defaults to 1e-08):
        Value of epsilon for the Adam optimizer.
    sizeWindow (`int`, *optional*, defaults to 20480):
        Number of frames to consider at each batch.
    nEpoch (`int`, *optional*, defaults to 200):
        Number of epochs to run.
    samplingType (`str`, *optional*, defaults to 'samespeaker'):
        How to sample the negative examples in the CPC loss.
        Choose from: ['samespeaker', 'uniform', 'samesequence', 'sequential']
    nLevelsPhone (`int`, *optional*, defaults to 1):
        (Supervised mode only). Number of layers in the phone classification network.
    cpc_mode (`str`, *optional*, defaults to None):
        Some variations on CPC.
        Choose from: ['reverse', 'none']
    encoder_type (`str`, *optional*, defaults to 'cpc'):
        Replace the encoder network by mfcc features or learned filter banks.
        Choose from: ['cpc', 'mfcc', 'lfb']
    normMode (`str`, *optional*, defaults to 'layerNorm'):
        Type of normalization to use in the encoder network (default is layerNorm).
        Choose from: ['instanceNorm', 'ID', 'layerNorm', 'batchNorm']
    onEncoder (`bool`, *optional*, defaults to False):
        (Supervised mode only) Perform the classification on the encoder's output.
    random_seed (`int`, *optional*, defaults to None):
        Set a specific random seed.
    arMode (`str`, *optional*, defaults to 'LSTM'):
        Architecture to use for the auto-regressive network (default is LSTM).
        Choose from: ['GRU', 'LSTM', 'RNN', 'no_ar', 'transformer']
    nLevelsGRU (`int`, *optional*, defaults to 1):
        Number of layers in the autoregressive network.
    rnnMode (`str`, *optional*, defaults to 'transformer'):
        Architecture to use for the prediction network.
        Choose from: ['transformer', 'RNN', 'LSTM', 'linear', 'ffd', 'conv4', 'conv8', 'conv12']
    dropout (`bool`, *optional*, defaults to False):
        Add a dropout layer at the output of the prediction network.
    abspos (`bool`, *optional*, defaults to False):
        If the prediction network is a transformer, activate to use absolute coordinates.
    """
    
    def __init__(self, **kwargs):
        _configurable_parameters = {
            'hiddenEncoder': {
                'default_value': 256,
                'type': int
            },
            'hiddenGar': {
                'default_value': 256,
                'type': int
            },
            'nPredicts': {
                'default_value': 12,
                'type': int
            },
            'negativeSamplingExt': {
                'default_value': 128,
                'type': int
            },
            'learningRate': {
                'default_value': 2e-4,
                'type': float
            },
            'schedulerStep': {
                'default_value': -1,
                'type': int
            },
            'schedulerRamp': {
                'default_value': None,
                'type': int
            },
            'beta1': {
                'default_value': 0.9,
                'type': float
            },
            'beta2': {
                'default_value': 0.999,
                'type': float
            },
            'epsilon': {
                'default_value': 1e-08,
                'type': float
            },
            'sizeWindow': {
                'default_value': 20480,
                'type': int
            },
            'nEpoch': {
                'default_value': 200,
                'type': int
            },
            'samplingType': {
                'default_value': 'samespeaker',
                'type': 'str',
                'choices': ['samespeaker', 'uniform', 'samesequence', 'sequential']
            },
            'nLevelsPhone': {
                'default_value': 1,
                'type': int
            },
            'cpc_mode': {
                'default_value': None,
                'type': str,
                'choices': ['reverse', 'none']
            },
            'encoder_type': {
                'default_value': 'cpc',
                'type': str,
                'choices': ['cpc', 'mfcc', 'lfb']
            },
            'normMode': {
                'default_value': 'layerNorm',
                'type': str,
                'choices': ['instanceNorm', 'ID', 'layerNorm', 'batchNorm']
            },
            'onEncoder': {
                'default_value': False,
                'type': bool
            },
            'random_seed': {
                'default_value': None,
                'type': int
            },
            'arMode': {
                'default_value': 'LSTM',
                'type': str,
                'choices': ['GRU', 'LSTM', 'RNN', 'no_ar', 'transformer']
            },
            'nLevelsGRU': {
                'default_value': 1,
                'type': int
            },
            'rnnMode': {
                'default_value': 'transformer',
                'type': str,
                'choices': ['transformer', 'RNN', 'LSTM', 'linear', 'ffd', 'conv4', 'conv8', 'conv12']
            },
            'dropout': {
                'default_value': False,
                'type': bool
            },
            'abspos': {
                'default_value': False,
                'type': bool
            }
        }
        for kw, arg in kwargs.items():
            assert kw in _configurable_parameters, f"'{kw}' is not a configurable parameter for {self.__class__.__name__}. \nPossible parameters: {list(_configurable_parameters.keys())}"
            assert type(arg) == _configurable_parameters[kw]['type'], f"'{kw}' parameter should be of type {_configurable_parameters[kw]['type']}"
            if kw in _configurable_parameters and 'choices' in _configurable_parameters[kw]:
                assert arg in _configurable_parameters[kw]['choices'], f"{kw} must be one of {_configurable_parameters[kw]['choices']}"
            setattr(self, kw, arg)
            del _configurable_parameters[kw]
        for remaining_param, param_dict in _configurable_parameters.items():
            setattr(self, remaining_param, param_dict['default_value'])
            
    def __repr__(self):
        return f'{self.__class__.__name__} ' +\
            str(self.__dict__).replace('{', '{\n ').replace(',', ',\n').replace('}', '\n}')