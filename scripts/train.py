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

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
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
WEIGHT_DECAY = 1e-6
LR = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 250
GRAD_CLIP_NORM = 1.0
NUM_WARMUP_STEPS = 500

# == Other cfg ==
MODEL_SAVE_DIR = "ckpts"
MODEL_LOAD_PATH = None
LOG_INTERVAL = 5  # Log every `LOG_INTERVAL` batch


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

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
        data_stats = convert_stats_from_np_to_torch(data_stats, device)

        train_ds = get_libero_dataset(
            hdf5_files=hdf5_files,
            obs_keys=obs_keys,
            split="train",
            seq_length=cfg["action_pred_horizon"],
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
            MODEL_SAVE_DIR, f"{ENV}_epoch_{epoch_idx}_{timestamp}_model.pth"
        )
        logger.info(f"Saving model checkpoint at {output_model_path}")

        ema.copy_to(diff_model.parameters())
        torch.save(diff_model.state_dict(), output_model_path)

        writer.close()
