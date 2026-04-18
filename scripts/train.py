# ---
# Generated: 2026-04-06 | claude-opus-4-6
# Prompt: Training loop for diffusion policy with PushT environment.
# Modifications:
#   2026-04-14 | Prompt: Save model config in checkpoint and support language
#               conditioning | Checkpoint now saves model_config alongside
#               state_dict so the model can be reconstructed at inference.
#               Added LANG_ENCODER_TYPE / LANG_PROJ_DIM / FREEZE_LANG_ENCODER
#               settings.  Optimizer filters out frozen parameters.
#               TASK_DESCRIPTION passed to forward when language encoder is active.
#   2026-04-14 | Prompt: Per-task descriptions from JSON with dropout |
#               Descriptions loaded from data/task_descriptions.json keyed by
#               TASK_KEY (and optional TASK_SUBTASK for LIBERO).  Passed to
#               dataset which returns a random description per sample.
#               LANG_DROPOUT_PROB drops language conditioning for some batches.
#   2026-04-14 | Prompt: Move lang dropout to dataset | Removed batch-level
#               language dropout from training loop; per-sample dropout is now
#               handled in PushTDataset.  Removed unused random import.
#   2026-04-14 | Prompt: Fix text encoder loading for resnet_only | task_description
#               is now only extracted from the batch and passed to forward when the
#               model has a language encoder, preventing a KeyError and avoiding
#               unnecessary text processing for resnet_only encoder type.
#   2026-04-15 | Prompt: Add description support for LIBERO | Refactored
#               description loading to support both flat lists (PushT) and
#               per-task dicts (LIBERO). Passes task_descriptions_by_task and
#               LANG_DROPOUT_PROB to get_libero_dataset so LIBERO samples
#               return a "description" key with random sampling and dropout.
#   2026-04-15 | Prompt: Read task_descriptions_path from env config | Replaced
#               hardcoded TASK_DESCRIPTIONS_PATH and TASK_KEY with
#               cfg["task_descriptions_path"] and cfg["task_descriptions_key"]
#               so the path is centralized in env_config.py.
#   2026-04-15 | Prompt: tqdm-safe logging and description debug log | Added
#               TqdmLoggingHandler so logger.info doesn't break progress bars.
#               Log sample descriptions from the first batch to verify they
#               reach the model.
#   2026-04-15 | Prompt: Save data_stats in checkpoint | Saved a numpy copy of
#               LIBERO normalization stats (data_stats_np) into the checkpoint so
#               inference can unnormalize actions without scanning HDF5 files.
#   2026-04-15 | Prompt: Replace encoder_type with vision/text encoder | Replaced
#               ENCODER_TYPE, PRETRAINED_MODEL, TEXT_PRETRAINED_MODEL with
#               VISION_ENCODER and TEXT_ENCODER. Updated DiffusionPolicy call
#               and description-loading guard to use new params.
# ---

import json

import torch
import torch.nn as nn
from diffusers import EMAModel, get_scheduler, DDPMScheduler
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from diffusion_policy.model.diffusion_policy import DiffusionPolicy
from diffusion_policy.env_config import get_env_config
from diffusion_policy.dataset.libero import (
    compute_stats_from_hdf5,
    convert_stats_from_np_to_torch,
)
import os
import time
from datetime import datetime
import argparse


class TqdmLoggingHandler(logging.StreamHandler):
    """Routes log output through tqdm.write so progress bars aren't broken."""

    def emit(self, record: logging.LogRecord) -> None:
        msg: str = self.format(record)
        tqdm.write(msg)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[TqdmLoggingHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, choices=["pusht", "libero"])
args = parser.parse_args()
if args.env is None:
    parser.error("--env is required. Choose from: pusht, libero")

# == Inputs ==
ENV = args.env
cfg = get_env_config(ENV)

# == Training  hyperparameters ==
DATASET_PATH = os.path.join("data", "pusht_cchi_v7_replay.zarr.zip")
MODEL_SAVE_DIR = "ckpts"
MODEL_LOAD_PATH = None

# Encoder settings
# Vision: model key from PRETRAINED_VISION_MODELS, or None for ResNet-18
#   Options: "clip-vit-b-16", "siglip-base-patch16-224", "siglip2-base-patch16-224",
#            "dinov2-small", "dinov2-base", "dinov2-large", None
VISION_ENCODER: str | None = None
# Text: model key for clip/siglip text encoder, "text" for standalone, or None for no language
#   Options: "clip-vit-b-16", "siglip-base-patch16-224", "siglip2-base-patch16-224",
#            "text", None
TEXT_ENCODER: str | None = None
LANG_PROJ_DIM = 256
FREEZE_VISION_ENCODER = False  # freeze pretrained vision encoder weights
FREEZE_TEXT_ENCODER = False  # freeze pretrained text encoder weights
LANG_DROPOUT_PROB = 0.01  # probability of dropping language conditioning per sample

# Training  hyperparameters
WEIGHT_DECAY = 1e-6
LR = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
GRAD_CLIP_NORM = 1.0
NUM_WARMUP_STEPS = 500

# == Other cfg ==
MODEL_SAVE_DIR = "ckpts"
MODEL_LOAD_PATH = None
LOG_INTERVAL = 5  # Log every `LOG_INTERVAL` batch
CHECKPOINT_INTERVAL = 25  # Save checkpoint every `CHECKPOINT_INTERVAL` epochs


def save_checkpoint(diff_model, ema, epoch_idx, save_dir, env_name, data_stats=None):
    """Save a model checkpoint, optionally applying EMA weights first."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_path = os.path.join(
        save_dir, f"{env_name}_epoch_{epoch_idx}_{timestamp}_model.pth"
    )
    logger.info(f"Saving model checkpoint at {output_model_path}")

    # Temporarily copy EMA weights into model, save, then restore
    ema.store(diff_model.parameters())
    ema.copy_to(diff_model.parameters())
    save_dict: dict = {
        "model_state_dict": diff_model.state_dict(),
        "model_config": diff_model.model_config,
    }
    if data_stats is not None:
        save_dict["data_stats"] = data_stats
    torch.save(save_dict, output_model_path)

    ema.restore(diff_model.parameters())

    return output_model_path


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # === Load task descriptions ===
    # task_descriptions: flat list for single-task envs (PushT)
    # task_descriptions_by_task: dict mapping task language to description
    #   paraphrases for multi-task envs (LIBERO)
    task_descriptions: list[str] = []
    task_descriptions_by_task: dict[str, list[str]] | None = None
    desc_path: str = cfg.get("task_descriptions_path", "")
    desc_key: str = cfg.get("task_descriptions_key", ENV)
    if TEXT_ENCODER is not None and desc_path and os.path.exists(desc_path):
        with open(desc_path, "r") as f:
            all_descriptions: dict = json.load(f)
        entry = all_descriptions.get(desc_key, [])
        if isinstance(entry, dict):
            # Multi-task: entry maps task language to description lists
            task_descriptions_by_task = entry if entry else None
        elif isinstance(entry, list):
            task_descriptions = entry
        n_descs: int = (
            len(task_descriptions_by_task)
            if task_descriptions_by_task
            else len(task_descriptions)
        )
        logger.info(f"Loaded {n_descs} description entries for {desc_key}")

    # === Data ===
    if ENV == "pusht":
        from diffusion_policy.dataset.pusht import PushTDataset

        train_ds = PushTDataset(
            dataset_path=cfg["dataset_path"],
            pred_horizon=cfg["action_pred_horizon"],
            obs_horizon=cfg["obs_horizon"],
            action_horizon=cfg["action_exec_horizon"],
        )
    elif ENV == "libero":
        from diffusion_policy.dataset.libero import (
            get_libero_dataset,
            get_hdf5_files_from_folders,
        )

        # Make correction to hyperparams:
        # Diffusion Policy uses a 6D rotation representation for action output
        # from the paper Zhou et al. "On the continuity of rotation representations in neural networks."
        # The observed orientation in state will be quaternion (demonstration uses axis-angle, which gets converted in preprocess_batch).
        cfg["image_key"] = "agentview_rgb"
        cfg["state_keys"] = ["ee_pos", "ee_ori", "gripper_states"]
        cfg["state_obs_dim"] = 3 + 4 + 2
        cfg["action_dim"] = 3 + 6 + 1

        hdf5_files = get_hdf5_files_from_folders(cfg["dataset_path"])
        obs_keys = [cfg["image_key"]] + cfg["state_keys"]

        # Compute stats for states and actions for normalization and denormalization purpose
        data_stats = compute_stats_from_hdf5(hdf5_files, cfg["state_keys"])
        # Keep a numpy copy for the checkpoint so inference can unnormalize
        # without needing access to the original HDF5 files.
        data_stats_np = {
            "obs": {k: dict(v) for k, v in data_stats["obs"].items()},
            "actions": {k: dict(v) for k, v in data_stats["actions"].items()},
        }
        data_stats = convert_stats_from_np_to_torch(data_stats, device)

        train_ds = get_libero_dataset(
            hdf5_files=hdf5_files,
            obs_keys=obs_keys,
            split=None,  # use all 50 demonstrations
            seq_length=cfg["action_pred_horizon"],
            task_descriptions=task_descriptions_by_task,
            lang_dropout_prob=LANG_DROPOUT_PROB,
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # === Model ===
    diff_model = (
        DiffusionPolicy(
            action_dim=cfg["action_dim"],
            state_obs_dim=cfg["state_obs_dim"],
            obs_horizon=cfg["obs_horizon"],
            diff_step_dim=128,
            down_dims=[512, 1024, 2048],
            vision_encoder=VISION_ENCODER,
            text_encoder=TEXT_ENCODER,
            lang_proj_dim=LANG_PROJ_DIM,
            freeze_vision_encoder=FREEZE_VISION_ENCODER,
            freeze_text_encoder=FREEZE_TEXT_ENCODER,
        )
        .float()
        .to(device)
    )

    # cosine noise scheduler and clip output to [-1,1]
    diff_noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # keep an exponential moving average of diff_model parameters.
    ema = EMAModel(parameters=diff_model.parameters(), power=0.75)

    # === Training Modules ====
    # Only optimize parameters that require gradients (frozen encoder excluded)
    trainable_params = [p for p in diff_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    # cosine schedule with linear warmup (step after each epoch)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=len(train_dl) * NUM_EPOCHS,
    )
    loss_fn = nn.MSELoss()
    writer = SummaryWriter()  # tensorboard logger

    # === Training (Interruptable with Ctrl+C) ===
    start_time = time.perf_counter()
    epoch_idx = 0
    train_batch_idx = 0
    diff_model.train()

    # Continue from checkpoint if provided
    if MODEL_LOAD_PATH is not None:
        logger.info(f"Loading model checkpoint {MODEL_LOAD_PATH}")
        checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device, weights_only=True)
        # Support both new format (dict with model_config) and legacy (bare state_dict)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            diff_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            diff_model.load_state_dict(checkpoint)

    try:
        tepoch_range = tqdm(range(NUM_EPOCHS), desc="Epoch")
        for epoch_idx in tepoch_range:
            train_dl_pbar = tqdm(
                train_dl, total=len(train_dl), desc="Batch", leave=False
            )
            for train_batch in train_dl_pbar:
                #  Data Preprocess
                if ENV == "pusht":
                    imgs = train_batch[cfg["image_key"]].to(device)
                    states = train_batch[cfg["state_keys"][0]].to(device)
                    actions = train_batch["actions"].to(device)
                    language = None
                elif ENV == "libero":
                    from diffusion_policy.dataset.libero import preprocess_libero_batch

                    train_batch = preprocess_libero_batch(
                        train_batch,
                        stats=data_stats,
                        device=device,
                        obs_horizon=cfg["obs_horizon"],
                        image_keys=[cfg["image_key"]],
                    )
                    imgs = train_batch["obs"][cfg["image_key"]]
                    states = torch.cat(
                        [train_batch["obs"][k] for k in cfg["state_keys"]],
                        dim=-1,
                    )
                    actions = train_batch["actions"]
                    language = train_batch["language"]

                # Diffusion Training:
                # 1. Sample noise
                noises = torch.randn(actions.shape, device=device)

                # 2. Sample diffusion step
                diff_steps = torch.randint(
                    low=0,
                    high=cfg["num_diffusion_steps"],
                    size=(BATCH_SIZE,),
                    device=device,
                )

                # 3. Add noise to actions
                noisy_actions = diff_noise_scheduler.add_noise(
                    actions, noises, diff_steps
                )

                # 4. Compute noise residual as loss
                task_desc: list[str] | None = (
                    list(train_batch["description"])
                    if diff_model.lang_encoder is not None
                    else None
                )
                pred_noises = diff_model(
                    noisy_actions,
                    diff_steps,
                    imgs,
                    state_obs_seq=states,
                    task_description=task_desc,
                )

                loss = loss_fn(pred_noises, noises)

                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(diff_model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                lr_scheduler.step()
                ema.step(diff_model.parameters())

                # Log
                loss_cpu = loss.item()
                if train_batch_idx % LOG_INTERVAL == 0:
                    writer.add_scalar(
                        "Loss/train_every_50_batch",
                        loss_cpu,
                        train_batch_idx,
                    )
                    train_dl_pbar.set_postfix(loss=loss_cpu)
                train_batch_idx += 1

            # Periodic checkpoint saving
            if (epoch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                _stats = data_stats_np if ENV == "libero" else None
                save_checkpoint(
                    diff_model, ema, epoch_idx, MODEL_SAVE_DIR, ENV, data_stats=_stats
                )

        end_time = time.perf_counter()
        logger.info(f"Training complete. Took {end_time - start_time:.2f}s")

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually.")
    finally:
        # Save model parameters (EMA) either when interrupted or training done.
        _stats = data_stats_np if ENV == "libero" else None
        save_checkpoint(
            diff_model, ema, epoch_idx, MODEL_SAVE_DIR, ENV, data_stats=_stats
        )
        writer.close()
