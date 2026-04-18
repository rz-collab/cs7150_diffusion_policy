# ---
# Generated: 2026-04-18 | claude-sonnet-4-6
# Prompt: Batch evaluation script that runs multiple model checkpoints on all seen
#         LIBERO-10 tasks and optionally unseen tasks, collecting per-task success
#         rates and writing CSV reports. Validate mode uses init states 0-19, test
#         mode uses 20-39; unseen evaluation is gated by --run-eval-on-unseen flag.
# Modifications:
#   2026-04-18 | Prompt: Add --max-episodes flag for quick smoke testing | Added
#               optional --max-episodes arg that caps init_state_idxs to the first
#               N states of the mode's range, so a 2-episode run can verify the
#               pipeline without waiting for all 20 episodes per task.
#   2026-04-18 | Prompt: Early exit on LIBERO success | Break out of the episode
#               loop immediately when reward == 1.0 since LIBERO uses sparse binary
#               rewards and continuing after success wastes time.
#   2026-04-18 | Prompt: Fix progress bar granularity after batching | Switched from
#               tqdm over batches to tqdm(total=n_episodes) with pbar.update(len(batch))
#               so the bar still ticks once per episode regardless of batch size.
#   2026-04-18 | Prompt: Add batched environment inference | Replaced run_episode
#               with run_batched_episodes that connects to N servers, resets them
#               in parallel via ThreadPoolExecutor, stacks observations into a
#               single (N, ...) batch for one model forward pass, and dispatches
#               actions back to all envs simultaneously. Added --num-envs arg;
#               servers are expected on consecutive ports from --zmq-address.
# ---

"""
Evaluate trained diffusion policy checkpoints on LIBERO tasks.

Usage:
    # Single env (1 server on port 5555):
    conda run -n libero python scripts/libero_env_server.py --env libero_10 --port 5555
    conda run -n diff_policy python scripts/evaluate.py validate --checkpoints-dir ckpts/eval/

    # Batched (4 servers on ports 5555-5558):
    for port in 5555 5556 5557 5558; do
        conda run -n libero python scripts/libero_env_server.py --env libero_10 --port $port &
    done
    conda run -n diff_policy python scripts/evaluate.py validate \\
        --checkpoints-dir ckpts/eval/ --num-envs 4

    # Test mode with unseen tasks (separate server on port 5560):
    conda run -n libero python scripts/libero_env_server.py --env libero_spatial --port 5560
    conda run -n diff_policy python scripts/evaluate.py test \\
        --checkpoints-dir ckpts/eval/ --run-eval-on-unseen \\
        --unseen-zmq-address tcp://localhost:5560
"""

import argparse
import csv
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import DDPMScheduler
from tqdm import tqdm

from diffusion_policy.env_config import get_env_config
from diffusion_policy.model.diffusion_policy import (
    DiffusionPolicy,
    _convert_legacy_model_config,
)
from diffusion_policy.remote_env import RemoteEnv
from diffusion_policy.util.normalization import normalize_data, unnormalize_data

# ---------------------------------------------------------------------------
# Unseen tasks for generalization evaluation.
# Each entry must match the format returned by env.get_tasks():
#   {"idx": int, "name": str, "description": str}
# The idx should be the task index on whichever libero server you point
# --unseen-zmq-address at. Start that server with the appropriate suite
# (e.g. --env libero_spatial) before running with --run-eval-on-unseen.
# ---------------------------------------------------------------------------
UNSEEN_TASKS: list[dict] = [
    # TODO: Fill in once user specifies which unseen tasks to evaluate on.
    # Example:
    # {"idx": 0, "name": "SCENE0_put_the_bowl_on_the_plate", "description": "put the bowl on the plate"},
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-pool worker helpers — must be top-level to avoid closure captures
# ---------------------------------------------------------------------------

def _reset_env(args: tuple) -> dict:
    """Reset one remote env. Called in a thread pool."""
    env, task_idx, init_state_idx = args
    return env.reset(task_idx=task_idx, init_state_idx=init_state_idx)


def _step_env(args: tuple) -> tuple[dict, float, bool, dict]:
    """Step one remote env. Called in a thread pool."""
    env, action = args
    return env.step(action)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    env_key: str,
    device: torch.device,
) -> tuple[DiffusionPolicy, dict, dict]:
    """Load model, model_config, and normalization stats from a checkpoint file."""
    checkpoint: dict = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model_config: dict = checkpoint["model_config"]
        state_dict: dict = checkpoint["model_state_dict"]
    else:
        cfg = get_env_config(env_key)
        model_config = {
            "action_dim": cfg["action_dim"],
            "state_obs_dim": cfg["state_obs_dim"],
            "obs_horizon": cfg["obs_horizon"],
            "diff_step_dim": 128,
            "down_dims": [512, 1024, 2048],
        }
        state_dict = checkpoint

    # LIBERO uses 6D rotation representation; override dims to match training
    if env_key == "libero":
        model_config["action_dim"] = 3 + 6 + 1  # pos + rot_6d + gripper
        model_config["state_obs_dim"] = 3 + 4 + 2  # ee_pos + quat + gripper

    model_config = _convert_legacy_model_config(model_config)

    model = DiffusionPolicy(**model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if "data_stats" in checkpoint:
        stats: dict = checkpoint["data_stats"]
    else:
        from diffusion_policy.dataset.libero import (
            compute_stats_from_hdf5,
            get_hdf5_files_from_folders,
        )

        cfg = get_env_config(env_key)
        hdf5_files = get_hdf5_files_from_folders(cfg["dataset_path"])
        stats = compute_stats_from_hdf5(hdf5_files, ["ee_pos", "ee_ori", "gripper_states"])

    return model, model_config, stats


# ---------------------------------------------------------------------------
# Observation / action processing
# ---------------------------------------------------------------------------

def preprocess_state_libero(obs: dict, stats: dict) -> np.ndarray:
    """Normalize LIBERO observation state to match training preprocessing.

    robot0_eef_quat is patched by the server to use gripper0_grip_site
    (matching the controller frame). T.mat2quat returns [x,y,z,w]; we reorder
    to [w,x,y,z] to match the axis_angle_to_quaternion convention used in training.
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


# ---------------------------------------------------------------------------
# Batched episode runner
# ---------------------------------------------------------------------------

def run_batched_episodes(
    envs: list[RemoteEnv],
    model: DiffusionPolicy,
    model_config: dict,
    stats: dict,
    cfg: dict,
    device: torch.device,
    task_idx: int,
    batch_init_states: list[int],
    task_description: Optional[str],
    noise_scheduler: DDPMScheduler,
    executor: ThreadPoolExecutor,
) -> list[bool]:
    """Run one episode per env in parallel, returning a success flag for each.

    All envs run the same task but different init states. Observations from all
    active envs are stacked into a single batch for each model forward pass, so
    GPU utilization scales with the number of envs rather than being fixed at 1.
    """
    N: int = len(envs)

    # Reset all envs simultaneously
    obs_list: list[dict] = list(executor.map(
        _reset_env,
        [(env, task_idx, init_state_idx) for env, init_state_idx in zip(envs, batch_init_states)],
    ))

    # Seed per-env observation buffers with the initial observation
    obs_images: list[deque] = [deque(maxlen=cfg["obs_horizon"]) for _ in range(N)]
    obs_states: list[deque] = [deque(maxlen=cfg["obs_horizon"]) for _ in range(N)]
    for i, obs in enumerate(obs_list):
        img: np.ndarray = obs[cfg["image_key"]]
        state: np.ndarray = preprocess_state_libero(obs, stats)
        for _ in range(cfg["obs_horizon"]):
            obs_images[i].append(img)
            obs_states[i].append(state)

    use_lang: bool = task_description is not None and model.lang_encoder is not None
    max_steps: int = cfg["max_steps"]
    done: list[bool] = [False] * N
    success: list[bool] = [False] * N
    step_counts: list[int] = [0] * N

    while not all(done):
        active: list[int] = [i for i, d in enumerate(done) if not d]
        B: int = len(active)

        # Stack observations from active envs into one batch
        images_np: np.ndarray = np.stack([
            np.moveaxis(np.stack(obs_images[i]), -1, 1) / 255.0
            for i in active
        ])  # (B, obs_h, 3, H, W)
        states_np: np.ndarray = np.stack([
            np.stack(obs_states[i]) for i in active
        ])  # (B, obs_h, state_dim)

        images = torch.from_numpy(images_np).float().to(device)
        states = torch.from_numpy(states_np).float().to(device)

        noisy_actions = torch.randn(
            (B, cfg["action_pred_horizon"], model_config["action_dim"]),
            device=device,
        )
        noise_scheduler.set_timesteps(cfg["num_diffusion_steps"])

        task_desc: Optional[list[str]] = [task_description] * B if use_lang else None

        with torch.no_grad():
            for t in noise_scheduler.timesteps:
                noise_pred = model(
                    noisy_actions,
                    torch.full((B,), t, device=device, dtype=torch.long),
                    images,
                    state_obs_seq=states,
                    task_description=task_desc,
                )
                noisy_actions = noise_scheduler.step(
                    noise_pred, t, noisy_actions
                ).prev_sample

        # Slice action exec window for each active env
        start: int = cfg["obs_horizon"] - 1
        end: int = start + cfg["action_exec_horizon"]
        action_seqs: list[np.ndarray] = [
            denormalize_actions_libero(noisy_actions.detach().cpu()[b], stats)[start:end]
            for b in range(B)
        ]

        # Execute action sequence, stepping all still-active envs in parallel each tick
        for step in range(cfg["action_exec_horizon"]):
            still_active: list[tuple[int, int]] = [
                (b, i) for b, i in enumerate(active) if not done[i]
            ]
            if not still_active:
                break

            step_results: list = list(executor.map(
                _step_env,
                [(envs[i], action_seqs[b][step]) for b, i in still_active],
            ))

            for (b, i), (obs, reward, env_done, _) in zip(still_active, step_results):
                if reward == 1.0:
                    success[i] = True
                    done[i] = True
                    continue

                step_counts[i] += 1
                if env_done or step_counts[i] >= max_steps:
                    done[i] = True
                    continue

                obs_images[i].append(obs[cfg["image_key"]])
                obs_states[i].append(preprocess_state_libero(obs, stats))

    return success


# ---------------------------------------------------------------------------
# Task-level evaluation loop
# ---------------------------------------------------------------------------

def evaluate_on_tasks(
    model: DiffusionPolicy,
    model_config: dict,
    stats: dict,
    envs: list[RemoteEnv],
    cfg: dict,
    device: torch.device,
    tasks: list[dict],
    init_state_idxs: range,
    noise_scheduler: DDPMScheduler,
    executor: ThreadPoolExecutor,
) -> dict[str, float]:
    """Evaluate model across all tasks. Returns {task_name: success_rate}."""
    results: dict[str, float] = {}
    N: int = len(envs)
    all_states: list[int] = list(init_state_idxs)

    # Split init states into batches of N (last batch may be smaller)
    batches: list[list[int]] = [
        all_states[i: i + N] for i in range(0, len(all_states), N)
    ]

    for task_info in tasks:
        task_idx: int = task_info["idx"]
        task_name: str = task_info["name"]
        task_description: Optional[str] = task_info.get("description")

        successes: int = 0

        with tqdm(total=len(all_states), desc=f"  {task_name[:40]}", leave=False) as pbar:
            for batch in batches:
                batch_results: list[bool] = run_batched_episodes(
                    envs=envs[: len(batch)],  # trim to actual batch size
                    model=model,
                    model_config=model_config,
                    stats=stats,
                    cfg=cfg,
                    device=device,
                    task_idx=task_idx,
                    batch_init_states=batch,
                    task_description=task_description,
                    noise_scheduler=noise_scheduler,
                    executor=executor,
                )
                successes += sum(batch_results)
                pbar.update(len(batch))

        rate: float = successes / len(all_states)
        results[task_name] = rate
        logger.info(f"  {task_name[:50]}: {rate:.2%} ({successes}/{len(all_states)})")

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(output_path: str, rows: list[dict], task_names: list[str]) -> None:
    """Write a results table to CSV with one row per model."""
    fieldnames: list[str] = ["model"] + task_names + ["avg_success_rate"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved results → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _make_addresses(base_address: str, n: int) -> list[str]:
    """Expand a base ZMQ address into n consecutive-port addresses.

    "tcp://localhost:5555" with n=4 → ["tcp://localhost:5555", ..., "tcp://localhost:5558"]
    """
    prefix, port_str = base_address.rsplit(":", 1)
    base_port: int = int(port_str)
    return [f"{prefix}:{base_port + i}" for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion policy checkpoints on LIBERO tasks"
    )
    parser.add_argument(
        "mode",
        choices=["validate", "test"],
        help="validate: init states 0–19; test: init states 20–39",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing .pth checkpoint files to evaluate",
    )
    parser.add_argument(
        "--run-eval-on-unseen",
        action="store_true",
        help="Also evaluate on unseen tasks (only active in test mode)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="libero",
        help="Environment config key (default: libero)",
    )
    parser.add_argument(
        "--zmq-address",
        type=str,
        default="tcp://localhost:5555",
        help="Base ZMQ address for seen-task servers. With --num-envs N, servers are "
             "expected on consecutive ports starting here (default: tcp://localhost:5555)",
    )
    parser.add_argument(
        "--unseen-zmq-address",
        type=str,
        default=None,
        metavar="ADDR",
        help="Base ZMQ address for unseen-task servers (defaults to --zmq-address). "
             "With --num-envs N, expects N servers on consecutive ports from this base.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel environments (default: 1). Requires N libero servers "
             "running on consecutive ports starting at --zmq-address. Each batch of N "
             "init states is processed with a single batched model forward pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        metavar="DIR",
        help="Directory to save CSV result files (default: eval_results/)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of init states evaluated per task (e.g. 2 for a quick smoke test). "
             "Defaults to the full range (20 for validate, 20 for test).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

    # Init state index ranges per mode
    if args.mode == "validate":
        init_state_idxs: range = range(0, 20)
        run_eval_on_unseen: bool = False
    else:
        init_state_idxs = range(20, 40)
        run_eval_on_unseen = args.run_eval_on_unseen

    if args.max_episodes is not None:
        init_state_idxs = range(
            init_state_idxs.start,
            min(init_state_idxs.stop, init_state_idxs.start + args.max_episodes),
        )

    cfg: dict = {**get_env_config(args.env), "zmq_address": args.zmq_address}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(
        f"Device: {device} | Mode: {args.mode} | "
        f"Envs: {args.num_envs} | Init states: {list(init_state_idxs)}"
    )

    ckpt_files: list[Path] = sorted(Path(args.checkpoints_dir).glob("*.pth"))
    if not ckpt_files:
        logger.error(f"No .pth files found in {args.checkpoints_dir}")
        return
    logger.info(f"Found {len(ckpt_files)} checkpoint(s): {[f.name for f in ckpt_files]}")

    os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    with ThreadPoolExecutor(max_workers=args.num_envs) as executor:

        # ------------------------------------------------------------------
        # Seen task evaluation
        # ------------------------------------------------------------------
        seen_addresses: list[str] = _make_addresses(args.zmq_address, args.num_envs)
        logger.info(f"Connecting to seen-task server(s): {seen_addresses}")
        seen_envs: list[RemoteEnv] = [RemoteEnv(address=addr) for addr in seen_addresses]

        available_tasks: list[dict] = seen_envs[0].get_tasks()
        logger.info(f"Tasks ({len(available_tasks)}): {[t['name'] for t in available_tasks]}")
        seen_task_names: list[str] = [t["name"] for t in available_tasks]

        seen_rows: list[dict] = []
        for ckpt_path in ckpt_files:
            logger.info(f"--- Evaluating (seen): {ckpt_path.name} ---")
            model, model_config, stats = load_model(str(ckpt_path), args.env, device)

            task_results: dict[str, float] = evaluate_on_tasks(
                model=model,
                model_config=model_config,
                stats=stats,
                envs=seen_envs,
                cfg=cfg,
                device=device,
                tasks=available_tasks,
                init_state_idxs=init_state_idxs,
                noise_scheduler=noise_scheduler,
                executor=executor,
            )

            avg: float = sum(task_results.values()) / len(task_results) if task_results else 0.0
            logger.info(f"{ckpt_path.name} — seen avg success: {avg:.2%}")
            seen_rows.append({"model": ckpt_path.name, **task_results, "avg_success_rate": avg})

        seen_csv: str = os.path.join(args.output_dir, f"seen_tasks_{args.mode}.csv")
        write_csv(seen_csv, seen_rows, seen_task_names)
        for env in seen_envs:
            env.close()

        # ------------------------------------------------------------------
        # Unseen task evaluation (test mode only, when flag is set)
        # ------------------------------------------------------------------
        if run_eval_on_unseen:
            if not UNSEEN_TASKS:
                logger.warning(
                    "UNSEEN_TASKS list is empty — skipping unseen evaluation. "
                    "Populate UNSEEN_TASKS at the top of evaluate.py with the task "
                    "indices and names from the unseen libero server."
                )
            else:
                unseen_base: str = args.unseen_zmq_address or args.zmq_address
                unseen_addresses: list[str] = _make_addresses(unseen_base, args.num_envs)
                logger.info(f"Connecting to unseen-task server(s): {unseen_addresses}")
                unseen_envs: list[RemoteEnv] = [
                    RemoteEnv(address=addr) for addr in unseen_addresses
                ]

                unseen_task_names: list[str] = [t["name"] for t in UNSEEN_TASKS]
                unseen_rows: list[dict] = []

                for ckpt_path in ckpt_files:
                    logger.info(f"--- Evaluating (unseen): {ckpt_path.name} ---")
                    model, model_config, stats = load_model(str(ckpt_path), args.env, device)

                    task_results = evaluate_on_tasks(
                        model=model,
                        model_config=model_config,
                        stats=stats,
                        envs=unseen_envs,
                        cfg=cfg,
                        device=device,
                        tasks=UNSEEN_TASKS,
                        init_state_idxs=init_state_idxs,
                        noise_scheduler=noise_scheduler,
                        executor=executor,
                    )

                    avg = sum(task_results.values()) / len(task_results) if task_results else 0.0
                    logger.info(f"{ckpt_path.name} — unseen avg success: {avg:.2%}")
                    unseen_rows.append(
                        {"model": ckpt_path.name, **task_results, "avg_success_rate": avg}
                    )

                unseen_csv: str = os.path.join(args.output_dir, f"unseen_tasks_{args.mode}.csv")
                write_csv(unseen_csv, unseen_rows, unseen_task_names)
                for env in unseen_envs:
                    env.close()


if __name__ == "__main__":
    main()
