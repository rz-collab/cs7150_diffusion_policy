



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
- tensorboard fix if you get the bug for no module found pkg_resources: https://github.com/Nerogar/OneTrainer/issues/1304

## Code Source

The code from this project comes mainly from two research papers.

### Visuomotor Policy Learning via Action Diffusion

The first paper is **Visuomotor Policy Learning via Action Diffusion**. A lot of the code comes from this paper or is reimplemented similar to the code they did. Most of the code we used comes from their notebook code.

Paper Website: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)

Notebook Code: [https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=4APZkqh336-M](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=4APZkqh336-M)

### LIBERO

The other Paper used is LIBERO which is a benchmark dataset used for imitation learning in many different tasks. Below are websites linked to it to help with implementation and reference.

Dataset Website: [https://libero-project.github.io/main.html](https://libero-project.github.io/main.html)