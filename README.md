# internal_tools
`internal_tools` provides functionality for extracting various kinds of model internals from (audio-processing) Transformer architectures. It currently supports extracting **time-aligned hidden state activations** from several audio encoder architectures, specifically:
- accessed from the HuggingFace Hub through the `transformers` library: [Wav2Vec2Model](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2#transformers.Wav2Vec2Model), [HubertModel](https://huggingface.co/docs/transformers/en/model_doc/hubert#transformers.HubertModel), [WavLMModel](https://huggingface.co/docs/transformers/en/model_doc/wavlm#transformers.WavLMModel)
- architectures included in `internal_tools.models`: [SpidR](https://github.com/facebookresearch/spidr), [CPC](https://github.com/facebookresearch/CPC_audio/), [MelHuBERT](https://github.com/nervjack2/MelHuBERT/tree/main), [BabyHuBERT](https://huggingface.co/MarvinLvn/BabyHuBERT)

## Examples
Using `AudioPreprocessor` and `AudioModelExtractor` we can extract activations in a unified way for different model architectures: <details><summary>View code</summary>
```python
from internal_tools import AudioPreprocessor, AudioModelExtractor

from transformers import AutoModel
import torch
import sf

# load a Wav2Vec2 model through the HuggingFace hub
w2v2_model = AutoModel.from_pretrained('facebook/wav2vec2-base')
w2v2_preprocessor = AudioPreprocessor.for_hf_model('facebook/wav2vec2-base')

# load a SpidR model through torch.hub
spidr_model = torch.hub.load("facebookresearch/spidr", "spidr_base")
spidr_preprocessor = AudioPreprocessor.for_spidr_model()

# load audio file and extract SpidR activations for a specified time segment 
audio, sr = sf.read("my_audio_file.wav")
inputs = spidr_preprocessor(
    example_audio,
    sampling_rate=sr,
    return_tensors="pt"
).input_values

spidr_model.eval()
extr = AudioModelExtractor(spidr_model)
with torch.no_grad():
    outputs = spidr_model(inputs)
all_activations = extr.get_activations(
    # segment start & end time in seconds
    between_times=[(0.2, 0.5)], 
    input_size=inputs.shape[-1]
)
```
</details><br>

We can also efficiently extract activations for a dataset of multiple audio files and accompanying annotations: <details><summary>View code</summary>

```python
from internal_tools.dataset_utils import get_annotated_audio_loader

from collections import defaultdict
import numpy as np

dl = get_annotated_audio_loader(
    # a dataframe with start_time and end_time columns
    my_audio_annotations, 
    # path to the directory with corresponding audio files
    my_audio_directory,
    batch_size=50
)
extr = AudioModelExtractor(spidr_model)

actv_indices = []
activations = defaultdict(list)

for b, batch_data in enumerate(dl):
    inputs = spidr_preprocessor(
        batch_data['audio_signal'],
        sampling_rate=batch_data['audio_sampling_rate'],
        padding=True,
        return_tensors="pt",
        padding_side="right"
    ).input_values
    
    with torch.no_grad():
        outputs = spidr_model(inputs)
        
    extracted_actv_indices, extracted_acts = \
    extr.get_indexed_segment_activations(
        segment_index=batch_data['annotation_indices'], 
        segment_start_times=batch_data['annotation_start_times'],
        segment_end_times=batch_data['annotation_end_times'],
        input_size=inputs.shape[-1], 
        pooling='mean'
    )

    actv_indices.extend(extracted_actv_indices)
    for layer in extracted_acts:
        activations[layer].extend(extracted_acts[layer])

# stack and reorder activations by the annotations dataframe index
for layer in activations.keys():
    activations[layer] = np.stack([activations[layer][i] for i in np.argsort(actv_indices)])
```
</details><br>

See the notebooks and scripts in the [`tutorials/`](https://github.com/mdhk/internal_tools/tree/main/tutorials) directory for more details and  examples.

## To use this repository
1. Clone it locally:
   ```
   git clone git@github.com:mdhk/internal_tools.git
   ```
2. Initialize a [_uv_](https://docs.astral.sh/uv/) environment (first install `uv` if you haven't yet):
   ```
   uv python install 3.12.11
   uv python pin 3.12.11
   uv venv 'it-env'
   ```
3. Activate the environment and install dependencies
   ```
   source it-env/bin/activate
   uv pip install -r pyproject.toml
   ```
   For running code in the [`tutorials/`](https://github.com/mdhk/internal_tools/tree/main/tutorials) directory, you may also want to install the optional dependencies defined in [pyproject.toml](https://github.com/mdhk/internal_tools/blob/main/pyproject.toml).
4. Install `internal_tools` as a package:
   ```
   uv pip install -e .
   ```