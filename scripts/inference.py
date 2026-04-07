# ---
# Generated: 2026-04-06 01:55 UTC
# Modified: 2026-04-06 03:00 UTC
# Model: claude-opus-4-6
# Prompt: Read the entire project and help me finish writing the inference module
#         for the diffusion policy PushT environment, including model loading,
#         DDPM denoising loop, observation buffering, and environment execution.
# Modification: Refactor to use shared env_config so environments are swappable
#               via --env flag (e.g. --env pusht, --env libero_spatial).
# ---

# TODO: Review the code generated and make sure it works properly

import argparse
import torch
import numpy as np
import os
import logging
from collections import deque
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion_policy import DiffusionPolicy
from diffusion_policy.dataset.pusht import PushTDataset, unnormalize_data, normalize_data
from diffusion_policy.env_config import get_env_config

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_LOAD_PATH = os.path.join("ckpts", "model.pth")  # update with actual checkpoint


# ---
# Generated: 2026-04-07 00:00 UTC
# Model: claude-opus-4-6
# Prompt: Add type annotations to all functions and variables in inference.py
# ---
def extract_state(obs: dict[str, np.ndarray], state_keys: list[str]) -> np.ndarray:
    """Extract and concatenate state values from observation dict."""
    parts: list[np.ndarray] = [np.asarray(obs[k]).flatten() for k in state_keys]
    return np.concatenate(parts)


def make_env(cfg: dict):
    """Create environment based on config gym_api type."""
    if cfg["gym_api"] == "gymnasium":
        import gymnasium as gym
        import gym_pusht  # noqa: F401 (registers the env)
        return gym.make(cfg["env_name"], render_mode="human")
    elif cfg["gym_api"] == "gym":
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        task_suite = benchmark.get_benchmark_dict()[cfg["env_name"]]()
        task = task_suite.get_task(0)
        bddl_file: str = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        return OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=cfg["image_size"],
            camera_widths=cfg["image_size"],
        )


def env_reset(env, cfg: dict) -> dict:
    """Reset env, returning obs dict regardless of gym API version."""
    if cfg["gym_api"] == "gymnasium":
        obs, info = env.reset()
    else:
        obs = env.reset()
    return obs


def env_step(env, action: np.ndarray, cfg: dict) -> tuple[dict, float, bool]:
    """Step env, returning (obs, reward, done) regardless of gym API version."""
    if cfg["gym_api"] == "gymnasium":
        obs, reward, terminated, truncated, info = env.step(action)
        done: bool = terminated or truncated
    else:
        obs, reward, done, info = env.step(action)
    return obs, reward, done


def run_inference(env_key: str = "pusht") -> None:
    cfg: dict = get_env_config(env_key)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}, environment: {env_key}")

    # === Load dataset for normalization stats ===
    dataset = PushTDataset(
        dataset_path=cfg["dataset_path"],
        pred_horizon=cfg["action_pred_horizon"],
        obs_horizon=cfg["obs_horizon"],
        action_horizon=cfg["action_exec_horizon"],
    )
    stats: dict = dataset.stats

    # === Load model ===
    diff_model = DiffusionPolicy(
        action_dim=cfg["action_dim"],
        state_obs_dim=cfg["state_obs_dim"],
        obs_horizon=cfg["obs_horizon"],
    ).to(device)

    checkpoint: dict = torch.load(MODEL_LOAD_PATH, map_location=device, weights_only=True)
    diff_model.load_state_dict(checkpoint)
    diff_model.eval()
    logger.info(f"Loaded checkpoint from {MODEL_LOAD_PATH}")

    # === Noise scheduler ===
    # Matches the code they have in their notebook
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # === Environment ===
    env = make_env(cfg)
    obs: dict = env_reset(env, cfg)

    obs_images = deque(maxlen=cfg["obs_horizon"])
    obs_states = deque(maxlen=cfg["obs_horizon"])

    # Seed the observation buffer by repeating the first observation
    img: np.ndarray = obs[cfg["image_key"]]
    state: np.ndarray = extract_state(obs, cfg["state_keys"])
    for _ in range(cfg["obs_horizon"]):
        obs_images.append(img)
        obs_states.append(state)

    rewards: list[float] = []
    step_idx: int = 0
    max_steps: int = cfg["max_steps"]

    done = False

    with tqdm(total=max_steps, desc="Inference") as pbar:
        while not done:
            # === Build observation tensors ===
            # TODO: It seems that images are being directly inputted into the model. Will checkm with Richard on this.
            # If images are being directly inputted into the model, then current code is ok to use and the todo can be removed.
            # Otherwise if images aren't directly being inputted, need to update the code so the images are correctly being
            # input into the model in the proper manner. Note that in the notebook they use a vision encoder and then input
            # the encodings into the model directly.
            images_np = np.stack(obs_images)          # (obs_h, H, W, 3)
            images_np = np.moveaxis(images_np, -1, 1)       # (obs_h, 3, H, W)
            images = torch.from_numpy(images_np).float().unsqueeze(0).to(device)

            states_np = np.stack(obs_states)           # (obs_h, state_dim)
            nstates: np.ndarray = normalize_data(states_np, stats["agent_pos"])
            states = torch.from_numpy(nstates).float().unsqueeze(0).to(device)

            # === DDPM denoising loop ===
            # 1 is used for the first since the "batch size" is 1
            noisy_actions = torch.randn(
                (1, cfg["action_pred_horizon"], cfg["action_dim"]), device=device
            )

            # Reset timesets to initial time to perform diffusion
            noise_scheduler.set_timesteps(cfg["num_diffusion_steps"])

            with torch.no_grad():
                for t in noise_scheduler.timesteps:
                    noise_pred = diff_model(
                        noisy_actions,
                        torch.full((1,), t, device=device, dtype=torch.long),
                        images,
                        state_obs_seq=states,
                    )
                    noisy_actions = noise_scheduler.step(
                        noise_pred, t, noisy_actions
                    ).prev_sample

            # === Denormalize predicted actions ===
            pred_actions = noisy_actions.detach().to('cpu').numpy()[0]
            pred_actions = unnormalize_data(pred_actions, stats["action"])

            # === Execute actions in environment ===
            # Performs actions up to action horizon which is specified in `diffusion_policy/env_config.py`
            for i in range(cfg["action_exec_horizon"]):
                obs, reward, done = env_step(env, pred_actions[i], cfg)
                # Save Rewards
                rewards.append(reward)
                # Save observation images and state
                obs_images.append(obs[cfg["image_key"]])
                obs_states.append(extract_state(obs, cfg["state_keys"]))

                # Update step index and progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=f"{reward:.3f}")

                if step_idx > max_steps:
                    done=True

                if done:
                    break

            if done:
                break

    env.close()
    logger.info(f"Total steps: {step_idx}, Total reward: {sum(rewards):.2f}")
    if rewards:
        logger.info(f"Max reward: {max(rewards):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion policy inference")
    parser.add_argument(
        "--env", type=str, default="pusht",
        help="Environment config key (default: pusht). See env_config.py for options.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (overrides default)",
    )
    args = parser.parse_args()
    if args.checkpoint:
        MODEL_LOAD_PATH = args.checkpoint
    run_inference(env_key=args.env)
