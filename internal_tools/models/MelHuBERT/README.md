# MelHuBERT model

The `model.py` and `module.py` files in this adapt code from the [MelHuBERT model repository](https://github.com/nervjack2/MelHuBERT), as released with the paper [MelHuBERT: A Simplified Hubert on Mel Spectrograms
](http://doi.org/10.1109/ASRU57964.2023.10389700) (ASRU 2023); the config `.yaml` files and `.npy` files are also taken from this repository. This code is MIT licensed, as found in the LICENSE file. The MelHuBERT implementation itself uses fairseq and pytorch code, for which licenses are included in the respective folders.

The `melhubert_utils.py` file contains code for handling MelHuBERT models for use within `internal_tools`.