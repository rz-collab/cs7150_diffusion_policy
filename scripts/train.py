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

# Training  hyperparameters
WEIGHT_DECAY = 1e-6
LR = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 500
GRAD_CLIP_NORM = 1.0
NUM_WARMUP_STEPS = 500

# Logger
LOG_INTERVAL = 1

# TODO: Haven't added validation loop
# TODO: Data batch need to be preprocessed: obs horizon truncation, normalize image by /255, switch channel axis from H,W,C to C,H,W

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # === Data ===
    train_ds = PushTDataset(
        dataset_path=DATASET_PATH,
        pred_horizon=ACTION_PRED_HORIZON,
        obs_horizon=OBS_HORIZON,
        action_horizon=ACTION_EXEC_HORIZON,
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
    optimizer = torch.optim.AdamW(
        params=diff_model.parameters(),
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
        diff_model.load_state_dict(
            torch.load(MODEL_LOAD_PATH, map_location=device, weights_only=True)
        )

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
        torch.save(diff_model.state_dict(), output_model_path)

        writer.close()
