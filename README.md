# internal_tools
`internal_tools` provides functionality for extracting various kinds of model internals from (audio-processing) Transformer architectures. It currently supports extracting **time-aligned hidden state activations** from several audio encoder architectures for which pre-trained  checkpoints are available from various sources, specifically:
- through the HuggingFace Hub: [Wav2Vec2Model](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2#transformers.Wav2Vec2Model), [HubertModel](https://huggingface.co/docs/transformers/en/model_doc/hubert#transformers.HubertModel), [WavLMModel](https://huggingface.co/docs/transformers/en/model_doc/wavlm#transformers.WavLMModel)
- through `torch.hub`: [SpidR](https://github.com/facebookresearch/spidr)

TODO:
- MelHuBERT
- CPC