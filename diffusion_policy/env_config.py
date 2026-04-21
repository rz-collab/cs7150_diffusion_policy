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
        "train_task_suite": "pusht",
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
        "dataset_path": "data/libero_abs",
        "task_descriptions_path": TASK_DESCRIPTIONS_PATH,
        "train_task_suite": "libero_10",
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
