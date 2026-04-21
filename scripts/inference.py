import argparse
import json
import random
import torch
import numpy as np
import os
import logging
from collections import deque
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion_policy import (
    DiffusionPolicy,
    _convert_legacy_model_config,
)
from diffusion_policy.util.normalization import unnormalize_data, normalize_data
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


def preprocess_state_libero(obs: dict, stats: dict) -> np.ndarray:
    """Normalize LIBERO observation state to match training preprocessing.

    robot0_eef_quat is patched by the server to use gripper0_grip_site
    (matching the controller frame; see libero_env_server._patch_eef_obs).
    T.mat2quat returns [x,y,z,w]; training used RotationUtils.axis_angle_to_quaternion
    which returns [w,x,y,z], so we reorder here.
    """
    ee_pos: np.ndarray = normalize_data(
        np.asarray(obs["robot0_eef_pos"]).flatten(), stats["obs"]["ee_pos"]
    )
    q = np.asarray(obs["robot0_eef_quat"]).flatten()  # [x,y,z,w] from T.mat2quat
    ee_quat: np.ndarray = np.array([q[3], q[0], q[1], q[2]])  # reorder to [w,x,y,z]
    gripper: np.ndarray = normalize_data(
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
        stats["obs"]["gripper_states"],
    )
    return np.concatenate([ee_pos, ee_quat, gripper])


def denormalize_actions_libero(pred_actions: torch.Tensor, stats: dict) -> np.ndarray:
    """Convert 10D model output back to 7D LIBERO env actions.

    Model outputs: [pos_norm(3), rot_6d(6), gripper_norm(1)]
    Env expects:   [pos(3), axis_angle(3), gripper(1)]
    """
    import diffusion_policy.util.rotation as RotationUtils

    pos_np: np.ndarray = unnormalize_data(
        pred_actions[..., :3].numpy(), stats["actions"]["ee_pos"]
    )
    gripper_np: np.ndarray = unnormalize_data(
        pred_actions[..., 9:].numpy(), stats["actions"]["gripper_states"]
    )

    rot_6d = pred_actions[..., 3:9]
    rot_matrix = RotationUtils.rotation_6d_to_matrix(rot_6d)
    rot_aa_np: np.ndarray = RotationUtils.matrix_to_axis_angle(rot_matrix).numpy()

    return np.concatenate([pos_np, rot_aa_np, gripper_np], axis=-1)


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


def run_inference(
    env_key: str = "pusht",
    output_video_path: str | None = None,
    task_description: str | None = None,
    task_idx: int | None = None,
    display: bool = False,
    suite: str | None = None,
) -> None:
    cfg: dict = get_env_config(env_key)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {device}, environment: {env_key}")

    # === Load model (before stats so shape errors surface quickly) ===
    checkpoint: dict = torch.load(
        MODEL_LOAD_PATH, map_location=device, weights_only=False
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

    # LIBERO training overrides action/state dims for the 6D rotation
    # representation (Zhou et al.).  The checkpoint's model_config may
    # still contain the raw env values, so apply the same overrides here.
    if env_key == "libero":
        model_config["action_dim"] = 3 + 6 + 1  # pos + rot_6d + gripper
        model_config["state_obs_dim"] = 3 + 4 + 2  # ee_pos + quat + gripper

    # Convert old encoder_type-based configs to vision/text_encoder style
    model_config = _convert_legacy_model_config(model_config)

    diff_model = DiffusionPolicy(**model_config).to(device)
    diff_model.load_state_dict(state_dict)
    diff_model.eval()
    logger.info(f"Loaded checkpoint from {MODEL_LOAD_PATH}")

    # === Load normalization stats ===
    if env_key == "libero":
        if "data_stats" in checkpoint:
            stats: dict = checkpoint["data_stats"]
            logger.info("Loaded normalization stats from checkpoint")
        else:
            from diffusion_policy.dataset.libero import (
                compute_stats_from_hdf5,
                get_hdf5_files_from_folders,
            )

            logger.info("No stats in checkpoint, computing from HDF5 files")
            hdf5_files = get_hdf5_files_from_folders(cfg["dataset_path"])
            stats = compute_stats_from_hdf5(
                hdf5_files, ["ee_pos", "ee_ori", "gripper_states"]
            )
    else:
        from diffusion_policy.dataset.pusht import PushTDataset

        dataset = PushTDataset(
            dataset_path=cfg["dataset_path"],
            pred_horizon=cfg["action_pred_horizon"],
            obs_horizon=cfg["obs_horizon"],
            action_horizon=cfg["action_exec_horizon"],
        )
        stats = dataset.stats

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

    # For LIBERO, query available tasks from the server and select one.
    # The task description is used for language-conditioned models.
    if env_key == "libero":
        # If a suite is specified, switch the server to it before querying tasks
        # so get_tasks() returns the task list for that suite.
        if suite is not None:
            env.reset(suite_name=suite)
        available_tasks: list[dict] = env.get_tasks()
        if task_idx is not None:
            matched: list[dict] = [t for t in available_tasks if t["idx"] == task_idx]
            selected: dict = matched[0] if matched else available_tasks[task_idx]
        else:
            selected = random.choice(available_tasks)
            task_idx = selected["idx"]
        if task_description is None:
            task_description = selected["description"]
        logger.info(f"Task {task_idx}: {task_description}")
        obs: dict = env.reset(task_idx=task_idx, suite_name=suite)
    else:
        # PushT: pick a random description from the JSON file
        desc_path: str = cfg.get("task_descriptions_path", "")
        desc_key: str = cfg.get("train_task_suite", env_key)
        if task_description is None and desc_path and os.path.exists(desc_path):
            with open(desc_path, "r") as f:
                all_descriptions: dict = json.load(f)
            descs = all_descriptions.get(desc_key, [])
            if isinstance(descs, list) and descs:
                task_description = random.choice(descs)
        obs = env_reset(env, cfg)

    if task_description is not None:
        logger.info(f"Task description: {task_description}")

    save_video: bool = output_video_path is not None
    screen = None
    vis_size: int = cfg.get("vis_size", 512)
    if display:
        import pygame

        pygame.init()
        pygame.display.set_caption("Diffusion Policy Inference")
        screen = pygame.display.set_mode((vis_size, vis_size))

    obs_images = deque(maxlen=cfg["obs_horizon"])
    obs_states = deque(maxlen=cfg["obs_horizon"])

    # Seed the observation buffer by repeating the first observation
    img: np.ndarray = obs[cfg["image_key"]]
    if env_key == "libero":
        state: np.ndarray = preprocess_state_libero(obs, stats)
    else:
        state = extract_state(obs, cfg["state_keys"])
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
            if env_key == "libero":
                images_np = images_np / 255.0
            images = torch.from_numpy(images_np).float().unsqueeze(0).to(device)

            # LIBERO states are already normalized in preprocess_state_libero;
            # PushT states need min-max normalization here.
            states_np = np.stack(obs_states)  # (obs_h, state_dim)
            if env_key != "libero":
                states_np = normalize_data(states_np, stats["agent_pos"])
            states = torch.from_numpy(states_np).float().unsqueeze(0).to(device)

            # === DDPM denoising loop ===
            # 1 is used for the first since the "batch size" is 1
            noisy_actions = torch.randn(
                (1, cfg["action_pred_horizon"], model_config["action_dim"]),
                device=device,
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
            if env_key == "libero":
                pred_actions = denormalize_actions_libero(
                    noisy_actions.detach().cpu()[0], stats
                )
            else:
                pred_actions = noisy_actions.detach().cpu().numpy()[0]
                pred_actions = unnormalize_data(pred_actions, stats["actions"])

            # Only take action horrizon number of actions
            start = cfg["obs_horizon"] - 1
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
                if env_key == "libero":
                    obs_states.append(preprocess_state_libero(obs, stats))
                else:
                    obs_states.append(extract_state(obs, cfg["state_keys"]))

                # Render frame for display / video
                if display or save_video:
                    render_img: np.ndarray = env.render()
                    if render_img is not None:
                        if save_video:
                            video_frames.append(render_img)
                        if display and screen is not None:
                            surf = pygame.surfarray.make_surface(
                                np.transpose(render_img, (1, 0, 2))
                            )
                            screen.blit(
                                pygame.transform.scale(surf, (vis_size, vis_size)),
                                (0, 0),
                            )
                            pygame.display.flip()
                if display:
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
    if save_video and video_frames:
        import imageio

        imageio.mimwrite(output_video_path, video_frames, fps=15)
        logger.info(f"Saved video ({len(video_frames)} frames) to {output_video_path}")

    env.close()
    if display:
        pygame.quit()
    logger.info(f"Total steps: {step_idx}, Total reward: {sum(rewards):.2f}")
    if rewards:
        logger.info(f"Max reward: {max(rewards):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diffusion policy inference")
    parser.add_argument(
        "--env",
        type=str,
        default="libero",
        help="Environment config key (default: libero). See env_config.py for options.",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides default)",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        metavar="PATH",
        help="Save output video to PATH (e.g. --save-video out.mp4)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live pygame window during inference",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Task description for language-conditioned models",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        help="LIBERO task index to run (random if omitted). "
        "Use get_tasks on the server to see available indices.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="LIBERO task suite to run (e.g. libero_spatial, libero_goal). "
        "Switches the server to that suite before selecting a task.",
    )
    args = parser.parse_args()
    if args.checkpoint:
        MODEL_LOAD_PATH = args.checkpoint
    run_inference(
        env_key=args.env,
        output_video_path=args.save_video,
        task_description=args.task_description,
        task_idx=args.task_idx,
        display=args.display,
        suite=args.suite,
    )
