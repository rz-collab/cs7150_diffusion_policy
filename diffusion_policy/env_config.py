# ---
# Generated: 2026-04-06 03:00 UTC
# Model: claude-opus-4-6
# Prompt: Create a shared environment config system so inference (and later training)
#         can switch between PushT, LIBERO, and other envs via a single flag.
# Modifications:
#   2026-04-14 | Prompt: Add ZMQ socket support for LIBERO | Added zmq_address field
#               to LIBERO configs so inference connects to the remote env server
#               instead of importing libero directly
#   2026-04-14 | Prompt: Make control_delta a changeable setting | Added control_delta
#               field to LIBERO configs (default True) so users can switch between
#               delta and absolute position action modes
#   2026-04-15 | Prompt: Centralize task_descriptions_path in env config | Added
#               task_descriptions_path field to each env config so train and
#               inference scripts read the path from config instead of hardcoding it.
#   2026-04-15 | Prompt: Switch LIBERO to absolute actions | Changed control_delta
#               from True to False so env config matches absolute-position mode
#               used by all LIBERO task suites.
# ---

# NOTE: This is a temporary file to integrate the two different environments frameworks.
# TODO: Test out the code to make sure it functions properly. Make sure loading the environment
# works properly and that its possible to change the environment, datasets, model parameters.
# Make sure that these are worth keeping or if there is anything I should change to make it
# better/function better.

# |o|o|                             observations: 2
# | |a|a|a|a|a|a|a|a|               actions executed: 8
# |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

import os

TASK_DESCRIPTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "dataset", "task_descriptions.json"
)

ENV_CONFIGS = {
    "pusht": {
        "env_name": "gym_pusht/PushT-v0",
        "action_dim": 2,
        "state_obs_dim": 2,
        "image_size": 96,
        "image_key": "pixels",
        "state_keys": ["agent_pos"],
        "gym_api": "gymnasium",  # 5-tuple (obs, reward, terminated, truncated, info)
        "dataset_path": "data/pusht_cchi_v7_replay.zarr.zip",
        "task_descriptions_path": TASK_DESCRIPTIONS_PATH,
        "task_descriptions_key": "pusht",
        "obs_horizon": 2,
        "action_exec_horizon": 8,
        "action_pred_horizon": 16,
        "num_diffusion_steps": 100,
        "max_steps": 300,
    },
    "libero": {
        "env_name": "libero",
        "action_dim": 7,
        "state_obs_dim": 9,
        "image_size": 128,
        "image_key": "agentview_image",
        "state_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "gym_api": "gym",  # 4-tuple (obs, reward, done, info)
        "zmq_address": "tcp://localhost:5555",
        "dataset_path": ["data/libero/libero_10"],
        "task_descriptions_path": TASK_DESCRIPTIONS_PATH,
        "task_descriptions_key": "libero_10",
        "obs_horizon": 2,
        "action_exec_horizon": 8,
        "action_pred_horizon": 16,
        "num_diffusion_steps": 100,
        "max_steps": 300,
        # True = actions are deltas from current pose; False = absolute target poses
        "control_delta": False,
    },
}


def get_env_config(env_key):
    if env_key not in ENV_CONFIGS:
        available = ", ".join(ENV_CONFIGS.keys())
        raise ValueError(f"Unknown environment '{env_key}'. Available: {available}")
    return ENV_CONFIGS[env_key]
