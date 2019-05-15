# dt2219-speech-project

# Data

`./download.sh` to download the (100 gb) .zip file of the dataset.

`python3 data_parse.py` to generate the (1 gb) comprezzed numpy object file ' with the following keys:

```python
import numpy as np

obj = np.load('mfcc_full_samples.npz')

samples = obj['samples']
info = obj['info']
```

samples contains an array of `info['n_samples'] tuples, where each tuple is a (id, mfcc) sample. 

info contains the following keys:

* `n_samples`: the number of data samples 
* `composed_to_id`: dictionary mapping a composer name to an id
* `id_to_composer`: dictionary mapping an id to a composer name
* `n_fft`: n_fft parameter used in computing the mffc
* `hop_length`: hop_lengths parameter used in computing the mfcc
