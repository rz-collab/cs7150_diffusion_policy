# ---
# Generated: 2026-04-06 | claude-opus-4-6
# Prompt: Inference module for the diffusion policy PushT environment, including
#         model loading, DDPM denoising loop, observation buffering, and
#         environment execution.
# Modifications:
#   2026-04-06 | Prompt: Refactor to use shared env_config | Refactored so environments
#               are swappable via --env flag (e.g. --env pusht, --env libero_spatial)
#   2026-04-07 | Prompt: Add type annotations | Added type annotations to all function
#               parameters, return types, and non-obvious variable declarations
#   2026-04-08 | Prompt: Fix PushT observation handling | Fixed inference to handle both
#               flat observation arrays and dict observations, with fallback to
#               env.render() for images
#   2026-04-07 | Prompt: Fix env visual rendering | Updated make_env to pass obs_type
#               from config to gym.make() so observations include pixels and agent_pos
#               as a dict
#   2026-04-07 | Prompt: Add visual display during inference | Added standalone pygame
#               display window for visual rendering while keeping render_mode=rgb_array
#               for correct observation capture. Renders env frames to a 512x512 window
#               each step
#   2026-04-08 | Prompt: Hardcode render_mode and obs_type into make_env | Moved
#               render_mode="rgb_array" and obs_type="pixels_agent_pos" from env config
#               into make_env since they are fixed inference requirements
#   2026-04-08 | Prompt: Remove dead extract functions | Removed extract_image and
#               extract_state_from_obs since obs is always a dict now. Replaced usages
#               with direct dict access and extract_state
#   2026-04-14 | Prompt: Use ZMQ sockets for LIBERO | Replaced direct LIBERO imports
#               in make_env with RemoteEnv ZMQ client so LIBERO runs in its own conda
#               env via a server, connected over a socket
#   2026-04-14 | Prompt: Support language-conditioned checkpoints | Checkpoint loading
#               now reads model_config to reconstruct the model (including language
#               encoder settings).  Supports both new and legacy checkpoint formats.
#               Task description passed to forward during denoising loop.
# ---

# TODO: Review the code generated and make sure it works properly

import argparse
import json
import torch
import numpy as np
import os
import logging
from collections import deque
from tqdm import tqdm
from diffusers import DDPMScheduler
import pygame
import imageio
from diffusion_policy.model.diffusion_policy import DiffusionPolicy
from diffusion_policy.dataset.pusht import (
    PushTDataset,
    unnormalize_data,
    normalize_data,
)
from diffusion_policy.env_config import get_env_config

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_LOAD_PATH = os.path.join("ckpts", "model.pth")  # update with actual checkpoint


def extract_state(obs: dict[str, np.ndarray], state_keys: list[str]) -> np.ndarray:
    """Extract and concatenate state values from observation dict."""
    parts: list[np.ndarray] = [np.asarray(obs[k]).flatten() for k in state_keys]
    return np.concatenate(parts)


def make_env(cfg: dict):
    """Create environment based on config gym_api type.

    For LIBERO environments (gym_api == "gym"), connects to a remote ZMQ server
    instead of importing libero directly. Start the server first:
        conda activate libero
        python scripts/libero_env_server.py --env <env_key>
    """
    if cfg["gym_api"] == "gymnasium":
        import gymnasium as gym
        import gym_pusht  # noqa: F401 (registers the env)

        return gym.make(
            cfg["env_name"],
            render_mode="rgb_array",
            obs_type="pixels_agent_pos",
        )
    elif cfg["gym_api"] == "gym":
        from diffusion_policy.remote_env import RemoteEnv

        address: str = cfg.get("zmq_address", "tcp://localhost:5555")
        logger.info(f"Connecting to remote LIBERO env at {address}")
        return RemoteEnv(address=address)


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


TASK_DESCRIPTIONS_PATH = os.path.join("data", "task_descriptions.json")


def run_inference(
    env_key: str = "pusht",
    output_video_path: str = "inference_output.mp4",
    task_description: str | None = None,
) -> None:
    cfg: dict = get_env_config(env_key)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}, environment: {env_key}")

    # If no explicit --task, use the first description from the JSON
    if task_description is None and os.path.exists(TASK_DESCRIPTIONS_PATH):
        with open(TASK_DESCRIPTIONS_PATH, "r") as f:
            all_descriptions: dict = json.load(f)
        descs = all_descriptions.get(env_key, [])
        if isinstance(descs, list) and descs:
            task_description = descs[0]
            logger.info(f"Using description from JSON: {task_description}")

    # === Load dataset for normalization stats ===
    dataset = PushTDataset(
        dataset_path=cfg["dataset_path"],
        pred_horizon=cfg["action_pred_horizon"],
        obs_horizon=cfg["obs_horizon"],
        action_horizon=cfg["action_exec_horizon"],
    )
    stats: dict = dataset.stats

    # === Load model ===
    checkpoint: dict = torch.load(
        MODEL_LOAD_PATH, map_location=device, weights_only=True
    )

    # Support both new format (dict with model_config) and legacy (bare state_dict)
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model_config: dict = checkpoint["model_config"]
        state_dict: dict = checkpoint["model_state_dict"]
        logger.info(f"Loaded model config from checkpoint: {model_config}")
    else:
        # Legacy checkpoint: no model_config, fall back to env config defaults
        model_config = {
            "action_dim": cfg["action_dim"],
            "state_obs_dim": cfg["state_obs_dim"],
            "obs_horizon": cfg["obs_horizon"],
            "diff_step_dim": 128,
            "down_dims": [512, 1024, 2048],
        }
        state_dict = checkpoint

    diff_model = DiffusionPolicy(**model_config).to(device)
    diff_model.load_state_dict(state_dict)
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

    vis_size: int = cfg.get("vis_size", 512)
    pygame.init()
    pygame.display.set_caption("Diffusion Policy Inference")
    screen = pygame.display.set_mode((vis_size, vis_size))

    obs_images = deque(maxlen=cfg["obs_horizon"])
    obs_states = deque(maxlen=cfg["obs_horizon"])

    # Seed the observation buffer by repeating the first observation
    img: np.ndarray = obs[cfg["image_key"]]
    state: np.ndarray = extract_state(obs, cfg["state_keys"])
    for _ in range(cfg["obs_horizon"]):
        obs_images.append(img)
        obs_states.append(state)

    rewards: list[float] = []
    video_frames: list[np.ndarray] = []
    step_idx: int = 0
    max_steps: int = cfg["max_steps"]

    done = False

    with tqdm(total=max_steps, desc="Inference") as pbar:
        while not done:
            # === Build observation tensors ===
            images_np = np.stack(obs_images)  # (obs_h, H, W, 3)
            images_np = np.moveaxis(images_np, -1, 1)  # (obs_h, 3, H, W)
            images = torch.from_numpy(images_np).float().unsqueeze(0).to(device)

            states_np = np.stack(obs_states)  # (obs_h, state_dim)
            nstates: np.ndarray = normalize_data(states_np, stats["agent_pos"])
            states = torch.from_numpy(nstates).float().unsqueeze(0).to(device)

            # === DDPM denoising loop ===
            # 1 is used for the first since the "batch size" is 1
            noisy_actions = torch.randn(
                (1, cfg["action_pred_horizon"], cfg["action_dim"]), device=device
            )

            # Reset timesets to initial time to perform diffusion
            noise_scheduler.set_timesteps(cfg["num_diffusion_steps"])

            # Build task description list for language conditioning
            task_desc: list[str] | None = None
            if task_description is not None and diff_model.lang_encoder is not None:
                task_desc = [task_description]

            with torch.no_grad():
                for t in noise_scheduler.timesteps:
                    noise_pred = diff_model(
                        noisy_actions,
                        torch.full((1,), t, device=device, dtype=torch.long),
                        images,
                        state_obs_seq=states,
                        task_description=task_desc,
                    )
                    noisy_actions = noise_scheduler.step(
                        noise_pred, t, noisy_actions
                    ).prev_sample

            # === Denormalize predicted actions ===
            pred_actions = noisy_actions.detach().to("cpu").numpy()[0]
            pred_actions = unnormalize_data(pred_actions, stats["action"])

            # Only take action horrizon number of actions
            start = cfg["action_exec_horizon"] - 1
            end = start + cfg["action_exec_horizon"]
            action = pred_actions[start:end, :]

            # === Execute actions in environment ===
            # Performs actions up to action horizon which is specified in `diffusion_policy/env_config.py`
            for i in range(len(action)):
                obs, reward, done = env_step(env, action[i], cfg)
                # Save Rewards
                rewards.append(reward)
                # Save observation images and state
                obs_images.append(obs[cfg["image_key"]])
                obs_states.append(extract_state(obs, cfg["state_keys"]))

                # Render frame to pygame display
                render_img: np.ndarray = env.render()
                if render_img is not None:
                    # Capture frame for video output
                    video_frames.append(render_img)
                    surf = pygame.surfarray.make_surface(
                        np.transpose(render_img, (1, 0, 2))
                    )
                    screen.blit(
                        pygame.transform.scale(surf, (vis_size, vis_size)), (0, 0)
                    )
                    pygame.display.flip()
                pygame.event.pump()

                # Update step index and progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=f"{reward:.3f}")

                if step_idx > max_steps:
                    done = True

                if done:
                    break

    # === Save video ===
    if video_frames:
        imageio.mimwrite(output_video_path, video_frames, fps=15)
        logger.info(f"Saved video ({len(video_frames)} frames) to {output_video_path}")

    env.close()
    pygame.quit()
    logger.info(f"Total steps: {step_idx}, Total reward: {sum(rewards):.2f}")
    if rewards:
        logger.info(f"Max reward: {max(rewards):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion policy inference")
    parser.add_argument(
        "--env",
        type=str,
        default="pusht",
        help="Environment config key (default: pusht). See env_config.py for options.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_output.mp4",
        help="Path to save output video (default: inference_output.mp4)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description for language-conditioned models",
    )
    args = parser.parse_args()
    if args.checkpoint:
        MODEL_LOAD_PATH = args.checkpoint
    run_inference(
        env_key=args.env,
        output_video_path=args.output,
        task_description=args.task,
    )
