# Exploring Generalization and Multi-Task Learning in Diffusion Policies with Language Conditioning

This repo contains the code for our final project in Deep Learning (CS 7150). The project adapts code from two papers, [Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) ([Repo](https://github.com/real-stanford/diffusion_policy)) and [LIBERO](https://libero-project.github.io/intro.html) ([Repo](https://github.com/Lifelong-Robot-Learning/LIBERO)). For more information on the sources of the code view [Code Source Section](#code-source)



## Installation

We will have two virtual environments, one for diffusion policy and one for LIBERO benchmarks, due to dependencies conflicts.

### Diffusion Policy (our package)
- Install torch with the appropriate CUDA version you have (check with `nvcc --version`)
- Install package and other dependencies.

```bash
conda create -n diff_policy python=3.13.12
conda activate diff_policy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### LIBERO Repo and Environment

Note: Optional, only used for inference using LIBERO simulation environment): We follow their exact installation instructions, which we repeat below for convenience.

First get the submodule downloaded.
```bash
git submodule update --init
```

Next is to get everything setup.
```bash
cd submodules/LIBERO
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

Now the repo should be fully setup and you are ready to use it with our code.

## Code Overview

```
cs7150_diffusion_policy/
├── diffusion_policy/                          # Main package (use: diff_policy env)
│   ├── env_config.py                          # Config registry for dims, paths, ZMQ address
│   ├── remote_env.py                          # ZMQ client for remote LIBERO environment
│   ├── dataset/
│   │   ├── pusht.py                           # PushT zarr dataset loader with language dropout
│   │   ├── libero.py                          # LIBERO HDF5 loader with normalization stats
│   │   └── task_descriptions.json             # Language descriptions for tasks
│   ├── model/
│   │   ├── diffusion_policy.py                # Main policy model assembly & checkpoint handling
│   │   ├── visual_encoder.py                  # Vision encoders: ResNet-18/CLIP/SigLIP/DINOv2
│   │   ├── language_encoder.py                # Text encoders: standalone or shared CLIP/SigLIP
│   │   ├── denoiser.py                        # Diffusion denoiser network
│   │   ├── encoder_base.py                    # Base ABC for freeze_backbone contract
│   │   └── diffusion_step_encoder.py          # Timestep encoding
│   └── util/
│       ├── normalization.py                   # Min-max normalization/denormalization
│       └── rotation.py                        # 6D rotation, axis-angle, quat, matrix conversions
│
├── scripts/                                   # Executable workflows
│   ├── train.py                               # Training entrypoint (diff_policy)
│   ├── inference.py                           # Rollout with optional language conditioning (diff_policy)
│   ├── evaluate.py                            # Batch evaluation across tasks (diff_policy) | multi-env, resumable CSV output
│   ├── libero_env_server.py                   # ZMQ server for LIBERO sim (libero env)
│   ├── rel2abs.py                             # Convert LIBERO HDF5 demos rel→abs actions (libero env) | 2 min/task, ~4+ hrs total
│   ├── compare_actions.py                     # Verify absolute action conversion (libero env) | smoke test
│   ├── test_env.py                            # Environment obs structure test (diff_policy)
│   ├── download_pusht_dataset.py              # Download PushT from Google Drive (diff_policy)
│   ├── pusht_env_xample.py                    # Minimal PushT random-action example (diff_policy)
│   ├── demo.ipynb                             # Example notebook workflow
│   ├── test_libero_control.ipynb              # LIBERO control testing
│   └── test_libero_abs_direct.ipynb           # Absolute action LIBERO testing
│
├── data/                                      # Create manually: mkdir data (download here)
│   ├── pusht/                                 # PushT zarr dataset (from download_pusht_dataset.py)
│   └── libero/                                # LIBERO HDF5 datasets (manual download + rel2abs.py)
│
├── runs/                                      # Created by train.py (tensorboard logs)
├── ckpts/                                     # Created by train.py (model checkpoints)
├── submodules/LIBERO/                         # git submodule for LIBERO benchmark
│
└── pyproject.toml                             # Package metadata
```

**Legend:**
- **(diff_policy)** = use `conda activate diff_policy` before running
- **(libero env)** = use `conda activate libero` before running
- Comment after script = key command-line arguments or purpose

## Datasets

There are two main datasets that we use for training our model: PushT and LIBERO.

PushT is a dataset we got from the original actions diffusion paper. Its a simple one to use and does not require the LIBERO dataset at all. However, we only use this dataset to prepare for LIBERO and make sure the model was functional before starting on the larger and more complicated LIBERO dataset. To setup and use follow instructions [here](#pusht-dataset-setup).

LIBERO is a dataset taken from the LIBERO paper and github. The authors created it to provide a dataset for imitation learning as well as test the models. LIBERO is the main dataset used for our paper, though does require setting up the LIBERO submodule as well as doing more. To setup and use follow instructions [here](#libero-datasets-setup).

### PushT Dataset Setup

Setup is very simple. Unless specified, the default environment is `diff_policy`.

All you need to do for setup is to follow the commands below. This will setup it all for you and no other actions are needed.
```bash
mkdir data
python scripts/download_pusht_dataset.py
```

### LIBERO Datasets Setup

Setting up LIBERO requires a lot more steps. First make sure you have installed the libero submodule and setup the environment using the LIBERO instructions in the [install section](#libero-repo-and-environment). Note that this entire repo assumes that you are using the `libero_10` dataset to train the model. This can be modified in the code, however, for simplicity we assume that is the dataset being used. In reality you would want to change the code to use `libero_90` as the training dataset and use the others for validation and testing.

#### Downloading Datasets

To then download the daataset you have two options.

1. Via HuggingFace Website: Download LIBERO datasets into `data/libero` folder: You can download them manually from `https://libero-project.github.io/datasets` or using their provided script (requires using `libero` conda environment), which downloads to `libero/datasets`.  Move them to `data/libero`.

2. Via CLI: To download the LIBERO dataset directly to the `data/libero` folder you can run the following script from inside the LIBERO environment: `python submodules/LIBERO/benchmark_scripts/download_libero_datasets.py --use-huggingface --download-dir data/libero --datasets all`.

Next is to prepare the dataset for the LIBERO model. While you can utalize the dataset as is, in the diffusion paper they found that the model works much better when dealing with absolute positions rather than deltas.

#### Converting to Absolute Position

To make diffusion policy use actions = position control instead of velocity, must run this script. This takes unfortunately 2 minute per task, and we have 130 tasks... This script is adapted from https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py, you can see explanation here: https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md
```bash
conda activate libero
python scripts/rel2abs.py --input_dir data/libero/libero_10
```

TODO: Continue updating README here.

Next is to verify that the `rel2abs.py` has performed properly. Below is a quick script that verifies it works by simulating absolute actions on the env and compare original video with new video. 
```bash
conda activate libero
python scripts/compare_actions.py
```

#### Libero Dataset Notes
- They use `Panda` robot model that has 7 revolution joints (`joint_states` dimension is 7) and a gripper of 2 DoF (fingers positions but they're symmetric, so in action space it's just one dimension).
The action dimension is 7: (px,py,pz,rx,ry,rz,gripper) and they're relative pose command (called `OSC_POSE` controller type in robosuite)

## Training Models

Each of the models are simple to train. To adjust how the models train, inside the `scripts/train.py` file at the top you will see bolded variables. This defines the training parameters we used. The ones we modified for our training were `VISION_ENCODER`, `LANG_PROJ_DIM`, `FREEZE_VISION_ENCODER`, and `FREEZE_TEXT_ENCODER`. These define what encoders where used for vision and language and if they were frozen (no adjustments to the parameters). There are comments above with the available models/what you can input into the models.

You can configure more aspects of the training of the model inside `diffusion_policy/env_config.py`. Here it is specified where the datasets to train the model are, what environment it uses, and any other specifics about the environment and training setup. To train to match what we used for our paper, change nothing here. Otherwise if you want to do more with our training setup, you can adjust the parameters here, just be careful and make sure you know what you are changing. The keys should be sufficiently self-explainatory.

Below are the commands to train the models.

Train for PushT task.
```bash
python scripts/train.py --env pusht
```

Train for LIBERO task.
```bash
python scripts/train.py --env libero
```

### Visualizing the Training Process

You can visualize the train loss via tensorboard. Run the command below inside the diff_policy environment and will show you the training loss over the number of steps the model has taken over all training sessions.

```bash
tensorboard --logdir runs/
```

## Testing Models

There are several ways to test the models. Note there is a very distict way to test the models PushT and LIBERO. PushT requires no additional steps to use the following testing script. LIBERO environment on the other hand requires setting up a server so that the environment can be run and interact with the model.

### LIBERO Server

The script

### Inference

Libero simulation environment server
```bash
conda activate libero
python scripts/libero_env_server.py --env libero_goal --save-video
```

### Evaluation

There is a script for evaluating the performance of the models. This mainly has been tested for LIBERO. So be aware that PushT may have some bugs that we are not aware of.

To evaluate, there is a few parameters to know about. Most importantly you it is recommended you set multiple servers to run since otherwise it will take an extremely long time to evaluate the models performance. It has two main modes `validation` and `test`. Validation is to validate the performance of all of the models. This will run through all of `libero_10` tasks with unique starting indices.


## Known Issues
- tensorboard fix if you get the bug for no module found pkg_resources: https://github.com/Nerogar/OneTrainer/issues/1304

## Code Source

The code from this project comes mainly from two research papers.

### Visuomotor Policy Learning via Action Diffusion

The first paper is **Visuomotor Policy Learning via Action Diffusion**. A lot of the code comes from this paper or is reimplemented similar to the code they did. Most of the code we used comes from their notebook code.

Paper Website: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)

Notebook Code: [https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=4APZkqh336-M](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=4APZkqh336-M)

### LIBERO Paper

The other Paper used is LIBERO which is a benchmark dataset used for imitation learning in many different tasks. Below are websites linked to it to help with implementation and reference.

Dataset Website: [https://libero-project.github.io/main.html](https://libero-project.github.io/main.html)

#### Downloading LIBERO and Datasets
To download libero you run the commands as follows. First you need to add the submodule. The submodule while originates from their repo, its a fork with small changes we made to make it work better with our environment.

Add the submodule with the following command:
```bash
git submodule update --init
```

After you add it follow their instructions to install it as a seperate conda enviroment. After that install the datasets with the following command.
```bash
python submodules/LIBERO/benchmark_scripts/download_libero_datasets.py --use-huggingface
```