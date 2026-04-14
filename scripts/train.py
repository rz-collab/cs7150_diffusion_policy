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
from diffusion_policy.dataset.pusht import PushTDataset
import os
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# == Inputs ==
DATASET_PATH = os.path.join("data", "pusht_cchi_v7_replay.zarr.zip")
MODEL_SAVE_DIR = "ckpts"
MODEL_LOAD_PATH = None

# Model hyperparameters
# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
OBS_HORIZON = 2
ACTION_EXEC_HORIZON = 8
ACTION_PRED_HORIZON = 16
NUM_DIFFUSION_STEPS_IN_TRAINING = 100
ACTION_DIM = 2
STATE_OBS_DIM = 2

# Encoder setting
# "resnet_only"     — ResNet-18 vision, no language
# "clip_text"       — ResNet-18 vision + pretrained text encoder
# "clip_both"       — Pretrained vision + pretrained text encoder
# "resnet_and_text" — ResNet-18 vision + standalone text encoder
ENCODER_TYPE = "resnet_and_text"
PRETRAINED_MODEL = "clip-vit-b-32"  # "clip-vit-b-16", "siglip-base-patch16-224", "siglip2-base-patch16-224"
LANG_PROJ_DIM = 256
FREEZE_ENCODERS = True  # freeze pretrained vision/language encoder weights
TASK_DESCRIPTIONS_PATH = os.path.join("data", "task_descriptions.json")
TASK_KEY = "pusht"  # top-level key in task_descriptions.json
TASK_SUBTASK: str | None = None  # subtask key for nested configs (e.g. LIBERO)
LANG_DROPOUT_PROB = 0.1  # probability of dropping language conditioning per batch

# Training  hyperparameters
WEIGHT_DECAY = 1e-6
LR = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50
GRAD_CLIP_NORM = 1.0
NUM_WARMUP_STEPS = 500

# Logger
LOG_INTERVAL = 1

# TODO: Haven't added validation loop

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # === Load task descriptions ===
    task_descriptions: list[str] = []
    if ENCODER_TYPE != "resnet_only" and os.path.exists(TASK_DESCRIPTIONS_PATH):
        with open(TASK_DESCRIPTIONS_PATH, "r") as f:
            all_descriptions: dict = json.load(f)
        entry = all_descriptions.get(TASK_KEY, [])
        if isinstance(entry, dict) and TASK_SUBTASK is not None:
            task_descriptions = entry.get(TASK_SUBTASK, [])
        elif isinstance(entry, list):
            task_descriptions = entry
        logger.info(f"Loaded {len(task_descriptions)} descriptions for {TASK_KEY}")

    # === Data ===
    train_ds = PushTDataset(
        dataset_path=DATASET_PATH,
        pred_horizon=ACTION_PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_EXEC_HORIZON,
        descriptions=task_descriptions,
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
    diff_model = DiffusionPolicy(
        action_dim=ACTION_DIM,
        state_obs_dim=STATE_OBS_DIM,
        obs_horizon=OBS_HORIZON,
        diff_step_dim=128,
        down_dims=[512, 1024, 2048],
        encoder_type=ENCODER_TYPE,
        pretrained_model=PRETRAINED_MODEL,
        lang_proj_dim=LANG_PROJ_DIM,
        freeze_encoders=FREEZE_ENCODERS,
    ).to(device)

    # cosine noise scheduler and clip output to [-1,1]
    diff_noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_STEPS_IN_TRAINING,
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
                imgs = train_batch["image"][:, :OBS_HORIZON].to(device)
                states = None
                if "agent_pos" in train_batch.keys():
                    states = train_batch["agent_pos"][:, :OBS_HORIZON].to(device)
                actions = train_batch["action"][:, :ACTION_PRED_HORIZON].to(device)

                # Diffusion Training:
                # 1. Sample noise
                noises = torch.randn(actions.shape, device=device)

                # 2. Sample diffusion step
                diff_steps = torch.randint(
                    low=0,
                    high=NUM_DIFFUSION_STEPS_IN_TRAINING,
                    size=(BATCH_SIZE,),
                    device=device,
                )

                # 3. Add noise to actions
                noisy_actions = diff_noise_scheduler.add_noise(
                    actions, noises, diff_steps
                )

                # 4. Compute noise residual as loss
                pred_noises = diff_model(
                    noisy_actions,
                    diff_steps,
                    imgs,
                    state_obs_seq=states,
                    task_description=list(train_batch["description"]),
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

        end_time = time.perf_counter()
        logger.info(f"Training complete. Took {end_time - start_time:.2f}s")

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually.")
    finally:
        # Save model parameters (EMA) either when interrupted or training done.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_model_path = os.path.join(
            MODEL_SAVE_DIR, f"epoch_{epoch_idx}_{timestamp}_model.pth"
        )
        logger.info(f"Saving model checkpoint at {output_model_path}")

        ema.copy_to(diff_model.parameters())
        torch.save(
            {
                "model_state_dict": diff_model.state_dict(),
                "model_config": diff_model.model_config,
            },
            output_model_path,
        )

        writer.close()
