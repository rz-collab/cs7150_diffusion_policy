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

### Quick Start

We have simplified our code into a notebook, so you can test out one of our pretrained models. Now that you have everything install you are all set and ready to use `inference_demo.ipynb`. Just go to the notebook and follow the instructions and you will have a running version of our model.

## Code Overview

```
cs7150_diffusion_policy/
│── inference_demo.ipynb                       # Notebook to run inference on the best performing model
│
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
│   └── download_pusht_dataset.py              # Download PushT from Google Drive (diff_policy)
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

To download the dataset you have two options.

1. Via HuggingFace Website: Download LIBERO datasets into `data/libero` folder: You can download them manually from `https://libero-project.github.io/datasets` or using their provided script (requires using `libero` conda environment), which downloads to `libero/datasets`.  Move them to `data/libero`.

2. Via CLI: To download the LIBERO dataset directly to the `data/libero` folder you can run the following script from inside the LIBERO environment: `python submodules/LIBERO/benchmark_scripts/download_libero_datasets.py --use-huggingface --download-dir data/libero --datasets all`.

Next, prepare the dataset for our training pipeline. In Diffusion Policy, performance is typically better when training on absolute end-effector targets rather than delta actions.

#### Converting to Absolute Position

Run conversion in the `libero` environment. The script recursively scans all `*.hdf5` files in `--input_dir` and writes a mirrored `_abs` directory.

Recommended command (converts all suites inside `data/libero`):
```bash
conda activate libero
python scripts/rel2abs.py --input_dir data/libero
```

This creates:
- `data/libero_abs/libero_10/...`
- `data/libero_abs/libero_goal/...`
- `data/libero_abs/libero_object/...`
- `data/libero_abs/libero_spatial/...`

Important: training defaults (`diffusion_policy/env_config.py`) expect `dataset_path="data/libero_abs"` and `train_task_suite="libero_10"`.

#### Verifying Conversion

Use the comparison script to replay converted absolute actions and save a side-by-side video:
```bash
conda activate libero
python scripts/compare_actions.py
```

The output video is written to:
- `verify_side_by_side.mp4`

If you use a different task or file path, update the hardcoded values at the bottom of `scripts/compare_actions.py` before running.

#### Libero Dataset Notes
- They use `Panda` robot model that has 7 revolution joints (`joint_states` dimension is 7) and a gripper of 2 DoF (fingers positions but they're symmetric, so in action space it's just one dimension).
The action dimension is 7: (px,py,pz,rx,ry,rz,gripper) and they're relative pose command (called `OSC_POSE` controller type in robosuite)

## Training Models

Training parameters are split in scripts/train.py and diffusion_policy/env_config.py
- Adjusting training hyperparameters (declared as constants at the top) in scripts/train.py. e.g., `VISION_ENCODER`, `LANG_PROJ_DIM`, `FREEZE_VISION_ENCODER`, and `FREEZE_TEXT_ENCODER`.
- Adjust datasets to train the model are, what environment it uses, some other rarely changed hyperparams like observation horizon in  `diffusion_policy/env_config.py`. 

Below are the commands to train the models.

Train for PushT task.
```bash
python scripts/train.py --env pusht
```

Train for LIBERO task.
```bash
python scripts/train.py --env libero
```

Run the command below inside the diff_policy environment to view the training progress and loss.
```bash
tensorboard --logdir runs/
```

## Testing Models

There are several ways to test the models. PushT can run directly in the `diff_policy` environment. LIBERO requires a separate server process in the `libero` environment, then clients connect over ZMQ from `diff_policy`.

### LIBERO Server

Since LIBERO has different dependencies, run it as a server in the `libero` conda environment.

Start one server (default port `5555`):
```bash
conda activate libero
python scripts/libero_env_server.py
```

Start a specific suite or port:
```bash
conda activate libero
python scripts/libero_env_server.py --env libero_10 --port 5555
```

Add the argument `--save-video` to save a video of the task being completed.

### Inference

These are inference scripts which do not require the usage of the notebook (`inference_demo.ipynb`).

PushT inference (no server needed):
```bash
conda activate diff_policy
python scripts/inference.py --env pusht --checkpoint ckpts/model.pth --save-video outputs/pusht_rollout.mp4
```

LIBERO inference (server required):
```bash
conda activate libero
python scripts/libero_env_server.py --env libero_goal --port 5555 --save-video
```

In a second terminal:
```bash
conda activate diff_policy
python scripts/inference.py --env libero --checkpoint ckpts/model.pth --task-idx 0 --suite libero_goal
```

### Evaluation

The evaluation script is focused on LIBERO checkpoints and writes resumable CSV files.

Modes:
- `validate`: evaluates init states `0-19`
- `test`: evaluates init states `20-39`

Run faster validation with 4 (lower if environments keep failing) parallel servers:
```bash
conda activate libero
python scripts/libero_env_server.py --env libero_10 --num-server 4
```

Then run evaluation:
```bash
conda activate diff_policy
python scripts/evaluate.py validate --checkpoints-dir ckpts/eval --zmq-address tcp://localhost:5555 --num-envs 4
```

Run test mode (optional unseen-task evaluation):
```bash
conda activate diff_policy
python scripts/evaluate.py test --checkpoints-dir ckpts/eval --run-on-unseen --zmq-address tcp://localhost:5555 --num-envs 4
```

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