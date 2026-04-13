# ---
# Generated: 2026-04-06 03:00 UTC
# Model: claude-opus-4-6
# Prompt: Create a shared environment config system so inference (and later training)
#         can switch between PushT, LIBERO, and other envs via a single flag.
# ---

# NOTE: This is a temporary file to integrate the two different environments frameworks.
# TODO: Test out the code to make sure it functions properly. Make sure loading the environment
# works properly and that its possible to change the environment, datasets, model parameters.
# Make sure that these are worth keeping or if there is anything I should change to make it
# better/function better.

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
        "obs_horizon": 2,
        "action_exec_horizon": 8,
        "action_pred_horizon": 16,
        "num_diffusion_steps": 100,
        "max_steps": 300,
    },
    "libero_spatial": {
        "env_name": "libero_spatial",
        "action_dim": 7,
        "state_obs_dim": 8,
        "image_size": 128,
        "image_key": "agentview_image",
        "state_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "gym_api": "gym",  # 4-tuple (obs, reward, done, info)
        "dataset_path": "data/libero_spatial",
        "obs_horizon": 2,
        "action_exec_horizon": 8,
        "action_pred_horizon": 16,
        "num_diffusion_steps": 100,
        "max_steps": 300,
    },
}


def get_env_config(env_key):
    if env_key not in ENV_CONFIGS:
        available = ", ".join(ENV_CONFIGS.keys())
        raise ValueError(f"Unknown environment '{env_key}'. Available: {available}")
    return ENV_CONFIGS[env_key]
