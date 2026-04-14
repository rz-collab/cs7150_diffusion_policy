# ---
# Generated: 2026-04-14 | claude-opus-4-6
# Prompt: Test script that connects to an environment and runs random actions
#         to verify the env setup (including ZMQ bridge for LIBERO) works.
# Modifications:
# ---

"""
Test an environment by running random actions.

Verifies that the environment can be created, reset, stepped, and rendered.
For LIBERO environments, start the ZMQ server first:
    conda activate libero
    python scripts/libero_env_server.py --env libero_spatial

Usage:
    python scripts/test_env.py --env pusht --steps 50
    python scripts/test_env.py --env libero_spatial --steps 50
"""

import argparse
import logging

import numpy as np

from diffusion_policy.env_config import get_env_config

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def make_env(cfg: dict):
    """Create environment based on config gym_api type."""
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
        obs, _info = env.reset()
    else:
        obs = env.reset()
    return obs


def env_step(env, action: np.ndarray, cfg: dict) -> tuple[dict, float, bool]:
    """Step env, returning (obs, reward, done) regardless of gym API version."""
    if cfg["gym_api"] == "gymnasium":
        obs, reward, terminated, truncated, _info = env.step(action)
        done: bool = terminated or truncated
    else:
        obs, reward, done, _info = env.step(action)
    return obs, reward, done


def run_test(env_key: str, num_steps: int) -> None:
    """Run random actions in the environment and print observation shapes."""
    cfg: dict = get_env_config(env_key)
    logger.info(f"Testing environment: {env_key}")

    # --- Create env ---
    env = make_env(cfg)
    logger.info("Environment created")

    # --- Reset ---
    obs: dict = env_reset(env, cfg)
    logger.info("Environment reset")

    # Log observation keys and shapes
    logger.info("Observation keys:")
    for key, val in obs.items():
        val_arr = np.asarray(val)
        logger.info(f"  {key}: shape={val_arr.shape}, dtype={val_arr.dtype}")

    # Verify expected keys are present
    image_key: str = cfg["image_key"]
    assert image_key in obs, f"Expected image key '{image_key}' not in obs: {list(obs.keys())}"
    for sk in cfg["state_keys"]:
        assert sk in obs, f"Expected state key '{sk}' not in obs: {list(obs.keys())}"
    logger.info("All expected observation keys present")

    # --- Step with random actions ---
    action_dim: int = cfg["action_dim"]
    rewards: list[float] = []

    for step in range(num_steps):
        action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        obs, reward, done = env_step(env, action, cfg)
        rewards.append(reward)

        if step == 0:
            # Log shapes after first step to confirm they're consistent
            logger.info("Observation after first step:")
            for key, val in obs.items():
                val_arr = np.asarray(val)
                logger.info(f"  {key}: shape={val_arr.shape}, dtype={val_arr.dtype}")

        if done:
            logger.info(f"Environment done at step {step + 1}")
            break

    # --- Render ---
    frame = env.render()
    if frame is not None:
        frame_arr = np.asarray(frame)
        logger.info(f"Render frame: shape={frame_arr.shape}, dtype={frame_arr.dtype}")
    else:
        logger.warning("env.render() returned None")

    # --- Summary ---
    steps_run: int = len(rewards)
    logger.info(f"Completed {steps_run}/{num_steps} steps")
    logger.info(f"Reward — total: {sum(rewards):.3f}, min: {min(rewards):.3f}, max: {max(rewards):.3f}")

    # --- Cleanup ---
    env.close()
    logger.info("Environment closed. Test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test environment with random actions")
    parser.add_argument(
        "--env",
        type=str,
        default="pusht",
        help="Environment config key (default: pusht). See env_config.py for options.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of random steps to run (default: 50)",
    )
    args = parser.parse_args()
    run_test(env_key=args.env, num_steps=args.steps)
