
## Installation

We will have two virtual environments, one for diffusion policy and one for LIBERO benchmarks, due to dependencies conflicts.

- Diffusion Policy (our package)
    - Install torch with the appropriate CUDA version you have (check with `nvcc --version`) 
    - Install package and other dependencies. 
    ```bash
    conda create -n diff_policy python=3.13.12
    conda activate diff_policy
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -e .
    ```

- LIBERO (optional, only used for inference using LIBERO simulation environment): We follow their exact installation instructions, which we repeat below for convenience
    ```bash
    git submodule update --init
    cd submodules/LIBERO 

    conda create -n libero python=3.8.13
    conda activate libero
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    pip install -r requirements.txt
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -e .
    ```

## Scripts

Unless specified, the default environment is `diff_policy`

Download PushT demonstration dataset
```bash
mkdir data
python scripts/download_pusht_dataset.py
```

Download LIBERO datasets into `data/libero` folder: \
You can download them manually from `https://libero-project.github.io/datasets` or using their provided script (requires using `libero` conda environment), which downloads to `libero/datasets`.  Move them to `data/libero`.

To make diffusion policy use actions = position control instead of velocity, must run this script.  This takes unfortunately 2 minute per task, and we have 130 tasks... This script is adapted from https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py, you can see explanation here: https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md
```bash
conda activate libero
python scripts/rel2abs.py --input_dir data/libero/libero_10
```

A quick script that verifies it works by simulating absolute actions on the env and compare original video with new video.
```bash
conda activate libero
python scripts/compare_actions.py
```

Train
```bash
python scripts/train.py --env libero
python scripts/train.py --env pusht
```

Visualize train loss via tensorboard
```bash
tensorboard --logdir runs/
```

Inference with Libero

Libero simulation environment server
```bash
conda activate libero
python scripts/libero_env_server.py --env libero_goal --save-video
```

## blabal
python scripts/inference.py --save-video libero_test.mp4 --env libero --checkpoint ckpts/libero_epoch_19_20260417_151825_model.pth --task-idx 3



## Libero Dataset Notes
- They use `Panda` robot model that has 7 revolution joints (`joint_states` dimension is 7) and a gripper of 2 DoF (fingers positions but they're symmetric, so in action space it's just one dimension).
The action dimension is 7: (px,py,pz,rx,ry,rz,gripper) and they're relative pose command (called `OSC_POSE` controller type in robosuite)




## Notes
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

#### Downloading LIBERO and Datasets
To download libero you run the commands as follows. First you need to add the submodule.

Add the submodule with the following command:
```bash
git submodule update --init
```

After you add it follow their instructions to install it as a seperate conda enviroment. After that install the datasets with the following command.
```bash
python submodules/LIBERO/benchmark_scripts/download_libero_datasets.py --use-huggingface
```