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
    
    # Validate
    conda run -n diff_policy python scripts/evaluate.py validate \\
        --checkpoints-dir ckpts/eval/ --num-envs 4

    # Test
    onda run -n diff_policy python scripts/evaluate.py test \\
        --checkpoints-dir ckpts/eval/ --run-on-unseen --num-envs 4
"""

import argparse
import csv
import logging
import os
import queue as _queue
import threading
from collections import deque
from concurrent.futures import Future as _Future
from pathlib import Path
from typing import Callable, Optional

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
    # libero_goal tasks (indices from libero_suite_task_map)
    {
        "idx": 3,
        "name": "open_the_top_drawer_and_put_the_bowl_inside",
        "description": "open the top drawer and put the bowl inside",
        "suite_name": "libero_goal",
    },
    {
        "idx": 9,
        "name": "put_the_wine_bottle_on_the_rack",
        "description": "put the wine bottle on the rack",
        "suite_name": "libero_goal",
    },
    {
        "idx": 5,
        "name": "push_the_plate_to_the_front_of_the_stove",
        "description": "push the plate to the front of the stove",
        "suite_name": "libero_goal",
    },
    # libero_object tasks
    {
        "idx": 4,
        "name": "pick_up_the_ketchup_and_place_it_in_the_basket",
        "description": "pick up the ketchup and place it in the basket",
        "suite_name": "libero_object",
    },
    {
        "idx": 7,
        "name": "pick_up_the_milk_and_place_it_in_the_basket",
        "description": "pick up the milk and place it in the basket",
        "suite_name": "libero_object",
    },
    # libero_spatial tasks
    {
        "idx": 2,
        "name": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "description": "pick up the black bowl from table center and place it on the plate",
        "suite_name": "libero_spatial",
    },
    {
        "idx": 7,
        "name": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "description": "pick up the black bowl on the stove and place it on the plate",
        "suite_name": "libero_spatial",
    },
]

logger = logging.getLogger(__name__)


class _TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm so log lines don't corrupt
    any active progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Thread-safe env proxy
# ---------------------------------------------------------------------------


class _EnvWorker:
    """Thread-safe RemoteEnv proxy.

    ZMQ sockets must be created and used in the same thread. This class owns
    a RemoteEnv in a dedicated background thread and routes all calls through
    a queue, returning concurrent.futures.Future objects so callers can
    submit N requests and wait for all N results in parallel.
    """

    _STOP = object()

    def __init__(
        self,
        address: str,
        ping_timeout_ms: int = 60000,
        op_timeout_ms: int = 600000,
    ) -> None:
        self._address: str = address
        # Persistent health flag — set True by callers (run_episodes_queue) when
        # this worker stops responding. Once dead, the worker stays out of the
        # pool for the rest of the run instead of blocking every subsequent call.
        self.dead: bool = False
        self._q: _queue.Queue = _queue.Queue()
        self._ready: threading.Event = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._t = threading.Thread(
            target=self._run,
            args=(address, ping_timeout_ms, op_timeout_ms),
            daemon=True,
            name=f"EnvWorker-{address}",
        )
        self._t.start()
        # Non-blocking: caller must call wait_ready() to check connection status.

    def wait_ready(self, timeout_s: float) -> bool:
        """Wait up to timeout_s for the background thread to connect.

        Returns True if connected, False if the timeout expired. Raises
        RuntimeError if the connection attempt failed (e.g. server not running).
        Marks self.dead=True on any failure so the worker is skipped by
        run_episodes_queue.
        """
        connected: bool = self._ready.wait(timeout=timeout_s)
        if self._init_error is not None:
            self.dead = True
            raise RuntimeError(
                f"Failed to connect to env server at {self._address}: "
                f"{self._init_error!r}. "
                f"Check that a libero_env_server is running on that port."
            ) from self._init_error
        if not connected:
            self.dead = True
        return connected

    def _run(self, address: str, ping_timeout_ms: int, op_timeout_ms: int) -> None:
        try:
            env = RemoteEnv(
                address=address,
                ping_timeout_ms=ping_timeout_ms,
                op_timeout_ms=op_timeout_ms,
            )
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._q.get()
            if item is self._STOP:
                env.close()
                return
            method, args, kwargs, fut = item
            try:
                fut.set_result(getattr(env, method)(*args, **kwargs))
            except Exception as exc:
                fut.set_exception(exc)

    def _submit(self, method: str, *args, **kwargs) -> _Future:
        """Submit a call and return a Future — does not block."""
        fut: _Future = _Future()
        self._q.put((method, args, kwargs, fut))
        return fut

    def get_tasks(self) -> list[dict]:
        return self._submit("get_tasks").result(timeout=60.0)

    def reset(
        self,
        task_idx: Optional[int] = None,
        init_state_idx: Optional[int] = None,
        suite_name: Optional[str] = None,
    ) -> dict:
        return self._submit(
            "reset",
            task_idx=task_idx,
            init_state_idx=init_state_idx,
            suite_name=suite_name,
        ).result(timeout=600.0)

    def step(self, action: np.ndarray) -> tuple:
        return self._submit("step", action).result(timeout=60.0)

    def close(self) -> None:
        # Don't try to drive a dead worker — its thread is stuck in a broken
        # recv and would never see the STOP sentinel.
        if not self.dead:
            self._q.put(self._STOP)
            self._t.join(timeout=10.0)


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
        hdf5_files = get_hdf5_files_from_folders(
            [os.path.join(cfg["dataset_path"], cfg["train_task_suite"])]
        )
        stats = compute_stats_from_hdf5(
            hdf5_files, ["ee_pos", "ee_ori", "gripper_states"]
        )

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
# Batched episode runner with work queue
# ---------------------------------------------------------------------------


def _seed_obs_buffers(
    obs: dict,
    obs_images: deque,
    obs_states: deque,
    cfg: dict,
    stats: dict,
) -> None:
    """Fill an env's obs buffers by repeating its initial observation."""
    img: np.ndarray = obs[cfg["image_key"]]
    state: np.ndarray = preprocess_state_libero(obs, stats)
    obs_images.clear()
    obs_states.clear()
    for _ in range(cfg["obs_horizon"]):
        obs_images.append(img)
        obs_states.append(state)


def run_episodes_queue(
    envs: list[_EnvWorker],
    model: DiffusionPolicy,
    model_config: dict,
    stats: dict,
    cfg: dict,
    device: torch.device,
    task_idx: int,
    all_init_states: list[int],
    task_description: Optional[str],
    suite_name: Optional[str],
    noise_scheduler: DDPMScheduler,
    pbar: Optional[tqdm] = None,
    step_timeout_s: float = 30.0,
    reset_timeout_s: float = 300.0,
) -> list[Optional[bool]]:
    """Run all init states using a work queue.

    N envs run in parallel. When an env finishes (success or timeout) it
    immediately resets to the next init state from the queue rather than waiting
    for its batch-mates. Batch size stays at N for the whole task, only dropping
    below N at the very end when the queue empties.

    Fault tolerance: if a server stops responding, the corresponding env is
    marked dead (envs[i].dead = True) and its in-flight init_state is returned
    to the queue so a surviving env can retry it. A dead env stays out of the
    pool for all subsequent tasks. If every live env dies before a retry
    succeeds, affected episodes are reported as ``None`` in the returned list
    so the caller can exclude them from the success-rate denominator rather
    than counting an env-side failure against the model.
    """
    N: int = len(envs)
    n_total: int = len(all_init_states)
    # None => episode never completed (excluded from the denominator upstream).
    results: list[Optional[bool]] = [None] * n_total

    # Queue of (result_idx, init_state_idx) — pop from end for O(1)
    queue: list[tuple[int, int]] = list(enumerate(all_init_states))
    queue.reverse()

    # Per-env tracking
    env_result_idx: list[Optional[int]] = [None] * N
    env_active: list[bool] = [False] * N
    env_dead: list[bool] = [e.dead for e in envs]
    obs_images: list[deque] = [deque(maxlen=cfg["obs_horizon"]) for _ in range(N)]
    obs_states: list[deque] = [deque(maxlen=cfg["obs_horizon"]) for _ in range(N)]
    step_counts: list[int] = [0] * N
    finished: int = 0

    def finish_episode(env_idx: int, success: bool) -> None:
        """Record a real outcome (policy succeeded or policy failed)."""
        nonlocal finished
        ridx: Optional[int] = env_result_idx[env_idx]
        if ridx is None:
            return
        results[ridx] = success
        env_result_idx[env_idx] = None
        finished += 1
        if pbar is not None:
            pbar.update(1)

    def kill_env(env_idx: int, op: str, exc: BaseException) -> None:
        """Mark env dead and requeue its in-flight init_state for retry."""
        if env_dead[env_idx]:
            return
        env_dead[env_idx] = True
        env_active[env_idx] = False
        envs[env_idx].dead = True  # persist across tasks
        logger.warning(
            f"Env {env_idx} ({envs[env_idx]._address}) unresponsive on {op}: "
            f"{type(exc).__name__}: {exc}. Removing from rotation."
        )
        # Push the interrupted init_state back for another env to retry.
        ridx: Optional[int] = env_result_idx[env_idx]
        if ridx is not None:
            queue.append((ridx, all_init_states[ridx]))
            env_result_idx[env_idx] = None

    def gather_reset(items: list[tuple[int, int]]) -> None:
        """Submit resets and seed obs for the ones that respond in time."""
        if not items:
            return
        futs = [
            envs[i]._submit(
                "reset", task_idx=task_idx, init_state_idx=s, suite_name=suite_name
            )
            for i, s in items
        ]
        for (env_idx, _), fut in zip(items, futs):
            try:
                obs: dict = fut.result(timeout=reset_timeout_s)
            except Exception as exc:
                kill_env(env_idx, op="reset", exc=exc)
                continue
            _seed_obs_buffers(obs, obs_images[env_idx], obs_states[env_idx], cfg, stats)

    # Start the first batch of envs, skipping any already-dead ones
    initial: list[tuple[int, int]] = []
    for i in range(N):
        if env_dead[i]:
            continue
        if not queue:
            break
        result_idx, init_state = queue.pop()
        env_result_idx[i] = result_idx
        env_active[i] = True
        initial.append((i, init_state))

    if not initial:
        logger.error(
            f"No live envs available for task {task_idx}; all "
            f"{n_total} episodes skipped (excluded from success rate)."
        )
        return results

    gather_reset(initial)

    use_lang: bool = task_description is not None and model.lang_encoder is not None
    max_steps: int = cfg["max_steps"]

    while finished < n_total:
        active: list[int] = [i for i in range(N) if env_active[i] and not env_dead[i]]
        if not active:
            # Every env has died. Remaining result slots stay None so the
            # caller excludes them from the success-rate denominator.
            remaining: int = n_total - finished
            logger.error(
                f"All envs dead mid-task; {remaining} remaining "
                f"episode(s) skipped (excluded from success rate)."
            )
            break
        B: int = len(active)

        # Build batch from active envs
        images_np: np.ndarray = np.stack(
            [np.moveaxis(np.stack(obs_images[i]), -1, 1) / 255.0 for i in active]
        )  # (B, obs_h, 3, H, W)
        states_np: np.ndarray = np.stack(
            [np.stack(obs_states[i]) for i in active]
        )  # (B, obs_h, state_dim)

        images = torch.from_numpy(images_np).float().to(device)
        states = torch.from_numpy(states_np).float().to(device)

        noisy_actions = torch.randn(
            (B, cfg["action_pred_horizon"], model_config["action_dim"]), device=device
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

        start: int = cfg["obs_horizon"] - 1
        end: int = start + cfg["action_exec_horizon"]
        action_seqs: list[np.ndarray] = [
            denormalize_actions_libero(noisy_actions.detach().cpu()[b], stats)[
                start:end
            ]
            for b in range(B)
        ]

        # Track which envs finish during this action chunk
        episode_done: list[bool] = [False] * N

        for step in range(cfg["action_exec_horizon"]):
            still_active: list[tuple[int, int]] = [
                (b, i)
                for b, i in enumerate(active)
                if env_active[i] and not episode_done[i] and not env_dead[i]
            ]
            if not still_active:
                break

            step_futs = [
                envs[i]._submit("step", action_seqs[b][step]) for b, i in still_active
            ]
            for (_, i), fut in zip(still_active, step_futs):
                try:
                    obs, reward, env_done, _ = fut.result(timeout=step_timeout_s)
                except Exception as exc:
                    kill_env(i, op="step", exc=exc)
                    episode_done[i] = True
                    continue

                if reward == 1.0:
                    finish_episode(i, success=True)
                    episode_done[i] = True
                    continue

                step_counts[i] += 1
                if env_done or step_counts[i] >= max_steps:
                    finish_episode(i, success=False)
                    episode_done[i] = True
                    continue

                obs_images[i].append(obs[cfg["image_key"]])
                obs_states[i].append(preprocess_state_libero(obs, stats))

        # Envs that finished: assign next init state (only live ones)
        reassign: list[tuple[int, int]] = []  # (env_idx, next_init_state)
        for i in [i for i in active if episode_done[i] and not env_dead[i]]:
            if queue:
                result_idx, init_state = queue.pop()
                env_result_idx[i] = result_idx
                step_counts[i] = 0
                reassign.append((i, init_state))
            else:
                env_active[i] = False

        gather_reset(reassign)

    return results


# ---------------------------------------------------------------------------
# Task-level evaluation loop
# ---------------------------------------------------------------------------


def evaluate_on_tasks(
    model: DiffusionPolicy,
    model_config: dict,
    stats: dict,
    envs: list[_EnvWorker],
    cfg: dict,
    device: torch.device,
    tasks: list[dict],
    init_state_idxs: range,
    noise_scheduler: DDPMScheduler,
    partial_results: Optional[dict[str, dict[str, int]]] = None,
    on_task_done: Optional[Callable[[dict[str, dict[str, int]]], None]] = None,
) -> dict[str, dict[str, int]]:
    """Evaluate model across all tasks.

    Returns {task_name: {"successes": int, "attempts": int}} where attempts
    excludes episodes that were skipped because every live env died. The
    success rate is ``successes / attempts`` (caller responsibility).

    partial_results: tasks already evaluated in a previous run; those tasks
      are skipped and their counts carried forward.
    on_task_done: called with the full accumulated results dict after each
      task completes, so callers can write a partial CSV row mid-checkpoint.
    """
    results: dict[str, dict[str, int]] = (
        dict(partial_results) if partial_results else {}
    )
    all_states: list[int] = list(init_state_idxs)

    for task_info in tasks:
        task_idx: int = task_info["idx"]
        task_name: str = task_info["name"]
        task_description: Optional[str] = task_info.get("description")
        suite_name: Optional[str] = task_info.get("suite_name")

        if task_name in results:
            logger.info(f"  Skipping {task_name[:50]} (already evaluated).")
            continue

        with tqdm(
            total=len(all_states), desc=f"  {task_name[:40]}", leave=False
        ) as pbar:
            episode_results: list[Optional[bool]] = run_episodes_queue(
                envs=envs,
                model=model,
                model_config=model_config,
                stats=stats,
                cfg=cfg,
                device=device,
                task_idx=task_idx,
                all_init_states=all_states,
                task_description=task_description,
                suite_name=suite_name,
                noise_scheduler=noise_scheduler,
                pbar=pbar,
            )

        # Episodes that couldn't be measured (all live envs died before retry
        # succeeded) appear as None. Exclude them from attempts so the rate
        # reflects policy behavior, not infra failures.
        completed: list[bool] = [r for r in episode_results if r is not None]
        skipped: int = len(episode_results) - len(completed)
        successes: int = sum(completed)
        attempts: int = len(completed)
        rate: float = successes / attempts if attempts else 0.0
        results[task_name] = {"successes": successes, "attempts": attempts}
        if skipped:
            logger.warning(
                f"  {task_name[:50]}: {skipped} episode(s) skipped due to env failures"
            )
        logger.info(f"  {task_name[:50]}: {rate:.2%} ({successes}/{attempts})")

        if on_task_done is not None:
            on_task_done(results)

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def build_csv_row(
    model_name: str,
    task_results: dict[str, dict[str, int]],
    tasks: list[dict],
    init_states: str = "",
) -> tuple[dict, float]:
    """Flatten per-task counts into a CSV row and compute the avg success rate.

    Tasks absent from task_results emit empty strings for successes/attempts so
    partial rows can be written mid-checkpoint. avg_rate is computed over
    evaluated tasks only.
    """
    row: dict = {"model": model_name, "init_states": init_states}
    rate_sum: float = 0.0
    n_tasks: int = 0

    for col_idx, t in enumerate(tasks):
        task_name: str = t["name"]
        suite: str = t.get("suite_name", "")
        server_idx: int = t["idx"]
        if task_name in task_results:
            s: int = task_results[task_name]["successes"]
            a: int = task_results[task_name]["attempts"]
            row[f"{col_idx}_successes"] = s
            row[f"{col_idx}_attempts"] = a
            row[f"{col_idx}_suite"] = suite
            row[f"{col_idx}_task_name"] = task_name
            row[f"{col_idx}_idx"] = server_idx
            rate_sum += (s / a) if a else 0.0
            n_tasks += 1
        else:
            row[f"{col_idx}_successes"] = ""
            row[f"{col_idx}_attempts"] = ""
            row[f"{col_idx}_suite"] = suite
            row[f"{col_idx}_task_name"] = task_name
            row[f"{col_idx}_idx"] = server_idx

    avg_rate: float = rate_sum / n_tasks if n_tasks else 0.0
    row["avg_success_rate"] = avg_rate
    return row, avg_rate


def _read_raw_rows(csv_path: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Read an existing CSV and return (fieldnames, {model: row_dict})."""
    if not os.path.exists(csv_path):
        return [], {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames: list[str] = list(reader.fieldnames or [])
        row_map: dict[str, dict[str, str]] = {row["model"]: dict(row) for row in reader}
    return fieldnames, row_map


def _merge_fieldnames(existing: list[str], tasks: list[dict]) -> list[str]:
    """Union of existing CSV columns and new task columns.

    Preserves existing column order, appends new task columns before
    avg_success_rate, ensures model is first and avg_success_rate is last.
    """
    existing_set: set[str] = set(existing)
    new_task_cols: list[str] = [
        col
        for col_idx in range(len(tasks))
        for col in [f"{col_idx}_successes", f"{col_idx}_attempts", f"{col_idx}_suite", f"{col_idx}_task_name", f"{col_idx}_idx"]
        if col not in existing_set
    ]
    base: list[str] = [c for c in existing if c not in {"model", "init_states", "avg_success_rate"}]
    return ["model", "init_states"] + base + new_task_cols + ["avg_success_rate"]


def write_csv(
    output_path: str,
    rows: list[dict],
    tasks: list[dict],
    verbose: bool = True,
) -> None:
    """Write a results table to CSV with one row per model.

    Preserves columns from any existing CSV that aren't in the current task list
    and adds new columns for tasks not yet present. For each row, merges existing
    CSV data (old task columns) with in-memory data (current task columns).
    """
    existing_fieldnames, old_row_map = _read_raw_rows(output_path)
    fieldnames: list[str] = _merge_fieldnames(existing_fieldnames, tasks)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            model: str = row["model"]
            merged: dict = {**old_row_map.get(model, {}), **row}
            writer.writerow({k: merged.get(k, "") for k in fieldnames})
    if verbose:
        logger.info(f"Saved results → {output_path}")


def _load_progress(
    csv_path: str,
    tasks: list[dict],
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]]]:
    """Read a results CSV and split rows into completed and partial.

    completed[ckpt_name] — every task column is filled in.
    partial[ckpt_name]   — at least one task column is empty; only tasks that
                           have been evaluated appear in the inner dict.
    Task indices are read from the header via {idx}_successes columns and
    mapped back to task names using the provided tasks list.
    """
    completed: dict[str, dict[str, dict[str, int]]] = {}
    partial: dict[str, dict[str, dict[str, int]]] = {}

    if not os.path.exists(csv_path):
        return completed, partial

    # Match tasks by (name, suite) so column order changes between runs don't
    # invalidate existing progress.
    current_keys: set[tuple[str, str]] = {
        (t["name"], t.get("suite_name", "")) for t in tasks
    }
    n_current: int = len(tasks)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        idx_strs: list[str] = [
            fn[: -len("_successes")]
            for fn in (reader.fieldnames or [])
            if fn.endswith("_successes")
        ]
        for row in reader:
            ckpt_name: str = row["model"]
            task_results: dict[str, dict[str, int]] = {}
            is_complete: bool = True
            for idx_str in idx_strs:
                csv_task_name: str = row.get(f"{idx_str}_task_name", "")
                csv_suite: str = row.get(f"{idx_str}_suite", "")
                if (csv_task_name, csv_suite) not in current_keys:
                    continue
                s_str: str = row.get(f"{idx_str}_successes", "")
                a_str: str = row.get(f"{idx_str}_attempts", "")
                if s_str == "" or a_str == "":
                    is_complete = False
                else:
                    task_results[csv_task_name] = {
                        "successes": int(s_str),
                        "attempts": int(a_str),
                    }
            if is_complete and len(task_results) < n_current:
                is_complete = False
            if is_complete:
                completed[ckpt_name] = task_results
            else:
                partial[ckpt_name] = task_results

    return completed, partial


def _init_csv(output_path: str, tasks: list[dict], force: bool = False) -> None:
    """Create or expand the CSV header.

    If the file doesn't exist (or force=True), writes a fresh header with current
    task columns. If it already exists, adds any missing task columns while keeping
    existing columns and data intact.
    """
    if force or not os.path.exists(output_path):
        fieldnames: list[str] = ["model", "init_states"]
        for col_idx in range(len(tasks)):
            fieldnames.extend([f"{col_idx}_successes", f"{col_idx}_attempts", f"{col_idx}_suite", f"{col_idx}_task_name", f"{col_idx}_idx"])
        fieldnames.append("avg_success_rate")
        with open(output_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        return

    existing_fieldnames, old_row_map = _read_raw_rows(output_path)
    merged: list[str] = _merge_fieldnames(existing_fieldnames, tasks)
    if merged == existing_fieldnames:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged)
        writer.writeheader()
        for old_row in old_row_map.values():
            writer.writerow({k: old_row.get(k, "") for k in merged})


# ---------------------------------------------------------------------------
# Unseen task validation
# ---------------------------------------------------------------------------

def _validate_unseen_tasks(tasks: list[dict], env: "_EnvWorker") -> None:
    """Check that every UNSEEN_TASKS entry name matches the server's task name at that idx.

    Groups tasks by suite, switches the server to each suite once, then compares
    the hardcoded name against what the server reports for that task_idx.
    Raises ValueError listing all mismatches so the caller can fix UNSEEN_TASKS.
    """
    from collections import defaultdict

    by_suite: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_suite[t.get("suite_name", "")].append(t)

    mismatches: list[str] = []
    for suite_name, suite_tasks in by_suite.items():
        env.reset(suite_name=suite_name)
        server_tasks: list[dict] = env.get_tasks()
        server_name_by_idx: dict[int, str] = {st["idx"]: st["name"] for st in server_tasks}
        for t in suite_tasks:
            actual: Optional[str] = server_name_by_idx.get(t["idx"])
            if actual is None:
                mismatches.append(
                    f"  [{suite_name}] idx {t['idx']}: not found on server"
                    f" (expected '{t['name']}')"
                )
            elif actual != t["name"]:
                mismatches.append(
                    f"  [{suite_name}] idx {t['idx']}:"
                    f" expected '{t['name']}', server has '{actual}'"
                )

    if mismatches:
        raise ValueError(
            "UNSEEN_TASKS name mismatch(es) — update UNSEEN_TASKS to match the server:\n"
            + "\n".join(mismatches)
        )


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


def _make_env_workers(
    addresses: list[str],
    startup_timeout_s: float = 30.0,
) -> list[_EnvWorker]:
    """Start all env workers simultaneously and wait with a shared deadline.

    All background threads begin connecting at the same instant. We then call
    wait_ready() on each with whatever time remains before the shared deadline,
    so evaluation starts after at most startup_timeout_s regardless of how many
    servers are slow or absent. Workers that don't respond in time are marked
    dead and excluded from evaluation.
    """
    import time

    workers: list[_EnvWorker] = [_EnvWorker(address=addr) for addr in addresses]
    deadline: float = time.monotonic() + startup_timeout_s

    for w in workers:
        remaining: float = max(0.0, deadline - time.monotonic())
        try:
            ready: bool = w.wait_ready(remaining)
            if not ready:
                logger.warning(
                    f"Server {w._address} did not respond within "
                    f"{startup_timeout_s}s — skipping."
                )
        except RuntimeError as exc:
            logger.warning(str(exc))

    live: int = sum(1 for w in workers if not w.dead)
    logger.info(f"{live}/{len(workers)} server(s) ready.")
    if live == 0:
        raise RuntimeError("No env servers responded during startup.")
    return workers


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
        "--run-on-unseen",
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
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any existing CSV results and restart evaluation from scratch.",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        metavar="IDX",
        help="Evaluate only a single task (by index). If not specified, all tasks are evaluated.",
    )
    args = parser.parse_args()

    # Route logs through tqdm.write so they don't clobber active progress bars.
    _handler = _TqdmLoggingHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)

    # Init state index ranges per mode
    if args.mode == "validate":
        init_state_idxs: range = range(0, 20)
        run_eval_on_unseen: bool = False
    else:
        init_state_idxs = range(20, 40)
        run_eval_on_unseen = args.run_on_unseen

    if args.max_episodes is not None:
        init_state_idxs = range(
            init_state_idxs.start,
            min(init_state_idxs.stop, init_state_idxs.start + args.max_episodes),
        )

    init_states_str: str = f"{init_state_idxs.start}-{init_state_idxs.stop - 1}"

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
    logger.info(
        f"Found {len(ckpt_files)} checkpoint(s): {[f.name for f in ckpt_files]}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # ------------------------------------------------------------------
    # Seen task evaluation
    # ------------------------------------------------------------------
    seen_addresses: list[str] = _make_addresses(args.zmq_address, args.num_envs)
    logger.info(f"Connecting to seen-task server(s): {seen_addresses}")
    seen_envs: list[_EnvWorker] = _make_env_workers(seen_addresses)

    seen_suite: str = cfg["train_task_suite"]
    # Switch server to the correct suite before querying tasks so suite_name
    # in the response reflects the actual suite being evaluated.
    seen_envs[0].reset(suite_name=seen_suite)
    available_tasks: list[dict] = seen_envs[0].get_tasks()
    for t in available_tasks:
        t["suite_name"] = seen_suite

    # Filter to a single task if --task-idx is specified
    if args.task_idx is not None:
        if args.task_idx < 0 or args.task_idx >= len(available_tasks):
            logger.error(
                f"Invalid --task-idx {args.task_idx}; available tasks: 0-{len(available_tasks) - 1}"
            )
            return
        available_tasks = [available_tasks[args.task_idx]]

    logger.info(
        f"Tasks ({len(available_tasks)}): {[t['name'] for t in available_tasks]}"
    )


    seen_csv: str = os.path.join(args.output_dir, f"seen_tasks_{args.mode}.csv")

    if not args.restart and os.path.exists(seen_csv):
        seen_completed, seen_partial = _load_progress(seen_csv, available_tasks)
        logger.info(
            f"Resuming seen evaluation: {len(seen_completed)} checkpoint(s) already done."
        )
    else:
        seen_completed, seen_partial = {}, {}
        if args.restart:
            logger.info("--restart: ignoring previous seen-task results.")

    _init_csv(seen_csv, available_tasks, force=args.restart)

    seen_rows: list[dict] = [
        build_csv_row(
            p.name, seen_completed[p.name], available_tasks, init_states_str
        )[0]
        for p in ckpt_files
        if p.name in seen_completed
    ]

    for ckpt_path in ckpt_files:
        if ckpt_path.name in seen_completed:
            logger.info(f"Skipping {ckpt_path.name} (already evaluated).")
            continue

        logger.info(f"--- Evaluating (seen): {ckpt_path.name} ---")
        model, model_config, stats = load_model(str(ckpt_path), args.env, device)

        def _on_seen_task_done(
            current_results: dict[str, dict[str, int]],
            ckpt_name: str = ckpt_path.name,
        ) -> None:
            partial_row, _ = build_csv_row(
                ckpt_name, current_results, available_tasks, init_states_str
            )
            write_csv(
                seen_csv, seen_rows + [partial_row], available_tasks, verbose=False
            )

        task_results: dict[str, dict[str, int]] = evaluate_on_tasks(
            model=model,
            model_config=model_config,
            stats=stats,
            envs=seen_envs,
            cfg=cfg,
            device=device,
            tasks=available_tasks,
            init_state_idxs=init_state_idxs,
            noise_scheduler=noise_scheduler,
            partial_results=seen_partial.get(ckpt_path.name),
            on_task_done=_on_seen_task_done,
        )

        row, avg = build_csv_row(
            ckpt_path.name, task_results, available_tasks, init_states_str
        )
        logger.info(f"{ckpt_path.name} — seen avg success: {avg:.2%}")
        seen_rows.append(row)
        write_csv(seen_csv, seen_rows, available_tasks)

    # ------------------------------------------------------------------
    # Unseen task evaluation (test mode only, when flag is set)
    # Reuses the same env workers — the server switches suites on demand
    # via the suite_name field in each reset request.
    # ------------------------------------------------------------------
    if run_eval_on_unseen:
        if not UNSEEN_TASKS:
            logger.warning(
                "UNSEEN_TASKS list is empty — skipping unseen evaluation. "
                "Populate UNSEEN_TASKS at the top of evaluate.py."
            )
        else:
            unseen_csv: str = os.path.join(
                args.output_dir, f"unseen_tasks_{args.mode}.csv"
            )

            if not args.restart and os.path.exists(unseen_csv):
                unseen_completed, unseen_partial = _load_progress(unseen_csv, UNSEEN_TASKS)
                logger.info(
                    f"Resuming unseen evaluation: {len(unseen_completed)} checkpoint(s) already done."
                )
            else:
                unseen_completed, unseen_partial = {}, {}
                if args.restart:
                    logger.info("--restart: ignoring previous unseen-task results.")

            _init_csv(unseen_csv, UNSEEN_TASKS, force=args.restart)

            unseen_rows: list[dict] = [
                build_csv_row(
                    p.name,
                    unseen_completed[p.name],
                    UNSEEN_TASKS,
                    init_states_str,
                )[0]
                for p in ckpt_files
                if p.name in unseen_completed
            ]

            for ckpt_path in ckpt_files:
                if ckpt_path.name in unseen_completed:
                    logger.info(
                        f"Skipping {ckpt_path.name} (already evaluated, unseen)."
                    )
                    continue

                logger.info(f"--- Evaluating (unseen): {ckpt_path.name} ---")
                model, model_config, stats = load_model(
                    str(ckpt_path), args.env, device
                )

                def _on_unseen_task_done(
                    current_results: dict[str, dict[str, int]],
                    ckpt_name: str = ckpt_path.name,
                ) -> None:
                    partial_row, _ = build_csv_row(
                        ckpt_name,
                        current_results,
                        UNSEEN_TASKS,
                        init_states_str,
                    )
                    write_csv(
                        unseen_csv,
                        unseen_rows + [partial_row],
                        UNSEEN_TASKS,
                        verbose=False,
                    )

                task_results = evaluate_on_tasks(
                    model=model,
                    model_config=model_config,
                    stats=stats,
                    envs=seen_envs,
                    cfg=cfg,
                    device=device,
                    tasks=UNSEEN_TASKS,
                    init_state_idxs=init_state_idxs,
                    noise_scheduler=noise_scheduler,
                    partial_results=unseen_partial.get(ckpt_path.name),
                    on_task_done=_on_unseen_task_done,
                )

                row, avg = build_csv_row(
                    ckpt_path.name,
                    task_results,
                    UNSEEN_TASKS,
                    init_states_str,
                )
                logger.info(f"{ckpt_path.name} — unseen avg success: {avg:.2%}")
                unseen_rows.append(row)
                write_csv(unseen_csv, unseen_rows, UNSEEN_TASKS)

    for env in seen_envs:
        env.close()


if __name__ == "__main__":
    main()
