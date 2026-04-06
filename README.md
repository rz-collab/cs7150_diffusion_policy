



## Requirements

Install torch with the appropriate CUDA version (check with `nvcc --version`)  \
Install package and other dependencies. 
```
python -m venv .venv
source .venv/bin/activate   # in windows: source .venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Scripts

Download PushT demonstration dataset
```
wget -O data/pusht_cchi_v7_replay.zarr.zip "https://drive.google.com/uc?export=download&id=1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
```

Train
```
python scripts/train.py
```