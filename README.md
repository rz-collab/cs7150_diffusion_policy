



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
mkdir data
python scripts/download_pusht_dataset.py
```

Train
```
python scripts/train.py
```

Visualize train loss via tensorboard
```
tensorboard --logdir runs/


```

## TODO

- I just copied dataset stuff, should probably read through it and clean it up.
- Haven't written the inference script yet. could use DDIM from diffusers, check https://huggingface.co/learn/diffusion-course/unit2/2#faster-sampling-with-ddim