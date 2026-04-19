# ---
# Generated: 2026-04-18 | claude-sonnet-4-6
# Prompt: Batch evaluation script that runs multiple model checkpoints on all seen
#         LIBERO-10 tasks and optionally unseen tasks, collecting per-task success
#         rates and writing CSV reports. Validate mode uses init states 0-19, test
#         mode uses 20-39; unseen evaluation is gated by --run-eval-on-unseen flag.
# Modifications:
#   2026-04-19 | Prompt: Resume from partial results, progressive CSV writes, --restart flag |
#               Added _load_progress to detect completed/partial checkpoint rows from an
#               existing CSV (using empty-string sentinels). Added _init_csv to create the
#               file with headers before any checkpoint finishes. Extended build_csv_row with
#               all_task_names param so partial rows can emit empty strings for unevaluated
#               tasks. write_csv now tolerates missing keys and accepts verbose=False for
#               silent mid-run writes. evaluate_on_tasks gains partial_results (skip already-
#               done tasks) and on_task_done callback (write partial CSV row after each task).
#               main restructured: CSV created at startup, progress loaded/skipped before the
#               loop, row appended after each checkpoint, partial row written after each task.
#               Same logic applied to the unseen eval block. --restart flag to wipe progress.
#   2026-04-19 | Prompt: Add unseen tasks and suite switching support | Populated
#               UNSEEN_TASKS with 7 tasks from libero_goal (idx 3,5,9),
#               libero_object (idx 4,7), and libero_spatial (idx 2,7). Added
#               suite_name field to task dicts and threaded it through
#               _EnvWorker.reset(), run_episodes_queue(), and evaluate_on_tasks()
#               so the server can switch suites without needing separate processes.
#   2026-04-19 | Prompt: dataset_path is base dir; join with train_task_suite; inject
#               suite_name into seen tasks | load_model fallback now computes the HDF5
#               folder as os.path.join(cfg["dataset_path"], cfg["train_task_suite"]).
#               After fetching available_tasks from the server, suite_name is set to
#               cfg["train_task_suite"] on each task dict so seen-task resets switch
#               to the correct suite at evaluation start.
#   2026-04-19 | Prompt: Fix stray break halting seen-task loop | Removed erroneous `break` on line 1038 that caused the checkpoint loop to exit immediately without evaluating any checkpoints.
#   2026-04-19 | Prompt: Add suite and init_states columns to CSV | build_csv_row,
#               write_csv, and _init_csv now accept suite and init_states params
#               (e.g. "libero_10", "20-39") and write them as the second and third
#               CSV columns after model. Unseen CSV records the sorted unique suite
#               names from UNSEEN_TASKS. _load_progress unchanged (ignores non-task cols).
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
#   2026-04-18 | Prompt: Work-queue episode scheduling | Replaced fixed-batch loop with
#               a queue so envs that finish early immediately reset to the next init
#               state instead of waiting for batch-mates. Batch size stays at N for
#               the whole task rather than shrinking at the end.
#   2026-04-18 | Prompt: Fix ZMQ thread-safety crash | ZMQ sockets must live and be
#               used in the same thread. Replaced bare RemoteEnv+ThreadPoolExecutor
#               with _EnvWorker: each env gets a dedicated thread that owns its socket.
#               Work is submitted via a queue; callers get a Future back. Removed the
#               ThreadPoolExecutor entirely — _EnvWorker threads provide the parallelism.
#   2026-04-18 | Prompt: Add batched environment inference | Replaced run_episode
#               with run_batched_episodes that connects to N servers, resets them
#               in parallel via ThreadPoolExecutor, stacks observations into a
#               single (N, ...) batch for one model forward pass, and dispatches
#               actions back to all envs simultaneously. Added --num-envs arg;
#               servers are expected on consecutive ports from --zmq-address.
#   2026-04-18 | Prompt: Fail fast on unreachable env server | _EnvWorker.__init__
#               now blocks until the worker thread finishes RemoteEnv setup and
#               re-raises any connection error in the main thread. Previously a
#               failed ping killed the worker thread silently and the main loop
#               hung on reset futures that would never complete.
#   2026-04-18 | Prompt: Bump reset timeout for task switches | Use the new
#               RemoteEnv ping/op timeout split (ping 60s, ops 600s) and raise
#               the reset Future timeout from 120s to 600s. Task transitions
#               force the server to close+rebuild a LIBERO MuJoCo env which
#               can exceed the old budgets when several servers rebuild at once.
#   2026-04-18 | Prompt: Graceful env-death handling | run_episodes_queue now
#               tolerates unresponsive servers: each Future.result call is
#               wrapped so a timeout or exception marks that env dead (via a
#               persistent _EnvWorker.dead flag), counts its in-flight episode
#               as a failure, and lets the remaining live envs continue. Dead
#               envs stay out of the pool for all subsequent tasks. Timeouts
#               were tightened to step=30s / reset=300s so hangs are detected
#               faster now that the run no longer dies on them.
#   2026-04-18 | Prompt: Retry env-killed episodes, drop from denominator when
#               unretriable | run_episodes_queue now returns list[Optional[bool]]
#               where None means the episode never completed because every live
#               env died before retry could succeed. When an env dies mid-
#               episode its init_state is pushed back onto the work queue so a
#               surviving env retries it. evaluate_on_tasks computes the task
#               success rate over completed episodes only (None entries
#               excluded from both numerator and denominator) and logs how
#               many were skipped due to env failures.
#   2026-04-18 | Prompt: Stop log lines corrupting progress bars | Added
#               _TqdmLoggingHandler that routes records through tqdm.write,
#               installed via basicConfig(force=True) so warnings emitted
#               mid-task (e.g. env death) no longer break the active tqdm bar.
#   2026-04-18 | Prompt: Save successes/attempts per task | evaluate_on_tasks
#               now returns {task: {"successes", "attempts"}} instead of just
#               the rate. CSV columns changed to <task>_successes /
#               <task>_attempts pairs plus total_successes, total_attempts,
#               and avg_success_rate (mean of per-task rates, equal-weighted).
#               New build_csv_row helper flattens the counts for the writer.
#   2026-04-18 | Prompt: Parallel worker startup with shared deadline |
#               _EnvWorker.__init__ is now non-blocking; wait_ready(timeout_s)
#               does the blocking check. _make_env_workers starts all threads
#               simultaneously then calls wait_ready with a shared deadline so
#               total startup time = min(all_ready, startup_timeout_s=30s).
#               Workers that miss the deadline are marked dead and skipped;
#               evaluation starts immediately with whoever connected.
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

    # Test mode with unseen tasks — same server handles all suites via suite switching:
    conda run -n libero python scripts/libero_env_server.py --env libero_10 --port 5555
    conda run -n diff_policy python scripts/evaluate.py test \\
        --checkpoints-dir ckpts/eval/ --run-eval-on-unseen
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
    {"idx": 3, "name": "open_the_top_drawer_and_put_the_bowl_inside", "description": "open the top drawer and put the bowl inside", "suite_name": "libero_goal"},
    {"idx": 9, "name": "put_the_wine_bottle_on_the_rack", "description": "put the wine bottle on the rack", "suite_name": "libero_goal"},
    {"idx": 5, "name": "push_the_plate_to_the_front_of_the_stove", "description": "push the plate to the front of the stove", "suite_name": "libero_goal"},
    # libero_object tasks
    {"idx": 4, "name": "pick_up_the_ketchup_and_place_it_in_the_basket", "description": "pick up the ketchup and place it in the basket", "suite_name": "libero_object"},
    {"idx": 7, "name": "pick_up_the_milk_and_place_it_in_the_basket", "description": "pick up the milk and place it in the basket", "suite_name": "libero_object"},
    # libero_spatial tasks
    {"idx": 2, "name": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "description": "pick up the black bowl from table center and place it on the plate", "suite_name": "libero_spatial"},
    {"idx": 7, "name": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", "description": "pick up the black bowl on the stove and place it on the plate", "suite_name": "libero_spatial"},
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
            target=self._run, args=(address, ping_timeout_ms, op_timeout_ms),
            daemon=True, name=f"EnvWorker-{address}",
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

    def reset(self, task_idx: Optional[int] = None, init_state_idx: Optional[int] = None, suite_name: Optional[str] = None) -> dict:
        return self._submit("reset", task_idx=task_idx, init_state_idx=init_state_idx, suite_name=suite_name).result(timeout=600.0)

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
            envs[i]._submit("reset", task_idx=task_idx, init_state_idx=s, suite_name=suite_name)
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
        active: list[int] = [
            i for i in range(N) if env_active[i] and not env_dead[i]
        ]
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
            denormalize_actions_libero(noisy_actions.detach().cpu()[b], stats)[start:end]
            for b in range(B)
        ]

        # Track which envs finish during this action chunk
        episode_done: list[bool] = [False] * N

        for step in range(cfg["action_exec_horizon"]):
            still_active: list[tuple[int, int]] = [
                (b, i) for b, i in enumerate(active)
                if env_active[i] and not episode_done[i] and not env_dead[i]
            ]
            if not still_active:
                break

            step_futs = [
                envs[i]._submit("step", action_seqs[b][step])
                for b, i in still_active
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
    results: dict[str, dict[str, int]] = dict(partial_results) if partial_results else {}
    all_states: list[int] = list(init_state_idxs)

    for task_info in tasks:
        task_idx: int = task_info["idx"]
        task_name: str = task_info["name"]
        task_description: Optional[str] = task_info.get("description")
        suite_name: Optional[str] = task_info.get("suite_name")

        if task_name in results:
            logger.info(f"  Skipping {task_name[:50]} (already evaluated).")
            continue

        with tqdm(total=len(all_states), desc=f"  {task_name[:40]}", leave=False) as pbar:
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
        logger.info(
            f"  {task_name[:50]}: {rate:.2%} ({successes}/{attempts})"
        )

        if on_task_done is not None:
            on_task_done(results)

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def build_csv_row(
    model_name: str,
    task_results: dict[str, dict[str, int]],
    all_task_names: Optional[list[str]] = None,
    suite: str = "",
    init_states: str = "",
) -> tuple[dict, float]:
    """Flatten per-task counts into a CSV row and compute totals.

    If all_task_names is given, tasks absent from task_results emit empty
    strings so partial rows can be written mid-checkpoint. avg_rate and totals
    are computed over evaluated tasks only.
    """
    row: dict = {"model": model_name, "suite": suite, "init_states": init_states}
    total_successes: int = 0
    total_attempts: int = 0
    rate_sum: float = 0.0
    n_tasks: int = 0

    ordered: list[str] = all_task_names if all_task_names is not None else list(task_results.keys())
    for task_name in ordered:
        if task_name in task_results:
            s: int = task_results[task_name]["successes"]
            a: int = task_results[task_name]["attempts"]
            row[f"{task_name}_successes"] = s
            row[f"{task_name}_attempts"] = a
            total_successes += s
            total_attempts += a
            rate_sum += (s / a) if a else 0.0
            n_tasks += 1
        else:
            row[f"{task_name}_successes"] = ""
            row[f"{task_name}_attempts"] = ""

    avg_rate: float = rate_sum / n_tasks if n_tasks else 0.0
    row["total_successes"] = total_successes
    row["total_attempts"] = total_attempts
    row["avg_success_rate"] = avg_rate
    return row, avg_rate


def write_csv(
    output_path: str,
    rows: list[dict],
    task_names: list[str],
    verbose: bool = True,
) -> None:
    """Write a results table to CSV with one row per model.

    Columns: model, suite, init_states, then (<task>_successes, <task>_attempts)
    pairs per task, then total_successes, total_attempts, and avg_success_rate
    (mean of the per-task rates). Missing keys in a row are written as empty
    strings so partial rows (mid-checkpoint) are valid CSV.
    """
    fieldnames: list[str] = ["model", "suite", "init_states"]
    for t in task_names:
        fieldnames.extend([f"{t}_successes", f"{t}_attempts"])
    fieldnames.extend(["total_successes", "total_attempts", "avg_success_rate"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    if verbose:
        logger.info(f"Saved results → {output_path}")


def _load_progress(
    csv_path: str,
) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]]]:
    """Read a results CSV and split rows into completed and partial.

    completed[ckpt_name] — every task column is filled in.
    partial[ckpt_name]   — at least one task column is empty; only tasks that
                           have been evaluated appear in the inner dict.
    Task names are inferred from the header via <name>_successes columns.
    """
    completed: dict[str, dict[str, dict[str, int]]] = {}
    partial: dict[str, dict[str, dict[str, int]]] = {}

    if not os.path.exists(csv_path):
        return completed, partial

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        task_names: list[str] = [
            fn[: -len("_successes")]
            for fn in (reader.fieldnames or [])
            if fn.endswith("_successes")
        ]
        for row in reader:
            ckpt_name: str = row["model"]
            task_results: dict[str, dict[str, int]] = {}
            is_complete: bool = True
            for task_name in task_names:
                s_str: str = row.get(f"{task_name}_successes", "")
                a_str: str = row.get(f"{task_name}_attempts", "")
                if s_str == "" or a_str == "":
                    is_complete = False
                else:
                    task_results[task_name] = {
                        "successes": int(s_str),
                        "attempts": int(a_str),
                    }
            if is_complete:
                completed[ckpt_name] = task_results
            else:
                partial[ckpt_name] = task_results

    return completed, partial


def _init_csv(output_path: str, task_names: list[str], force: bool = False) -> None:
    """Write the CSV header row. Skips if the file already exists unless force=True."""
    if not force and os.path.exists(output_path):
        return
    fieldnames: list[str] = ["model", "suite", "init_states"]
    for t in task_names:
        fieldnames.extend([f"{t}_successes", f"{t}_attempts"])
    fieldnames.extend(["total_successes", "total_attempts", "avg_success_rate"])
    with open(output_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


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
    logger.info(f"Found {len(ckpt_files)} checkpoint(s): {[f.name for f in ckpt_files]}")

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
    available_tasks: list[dict] = seen_envs[0].get_tasks()
    for t in available_tasks:
        t["suite_name"] = seen_suite
    logger.info(f"Tasks ({len(available_tasks)}): {[t['name'] for t in available_tasks]}")
    seen_task_names: list[str] = [t["name"] for t in available_tasks]

    seen_csv: str = os.path.join(args.output_dir, f"seen_tasks_{args.mode}.csv")

    if not args.restart and os.path.exists(seen_csv):
        seen_completed, seen_partial = _load_progress(seen_csv)
        logger.info(f"Resuming seen evaluation: {len(seen_completed)} checkpoint(s) already done.")
    else:
        seen_completed, seen_partial = {}, {}
        if args.restart:
            logger.info("--restart: ignoring previous seen-task results.")

    _init_csv(seen_csv, seen_task_names, force=args.restart)

    seen_rows: list[dict] = [
        build_csv_row(p.name, seen_completed[p.name], seen_task_names, seen_suite, init_states_str)[0]
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
            partial_row, _ = build_csv_row(ckpt_name, current_results, seen_task_names, seen_suite, init_states_str)
            write_csv(seen_csv, seen_rows + [partial_row], seen_task_names, verbose=False)

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

        row, avg = build_csv_row(ckpt_path.name, task_results, seen_task_names, seen_suite, init_states_str)
        logger.info(f"{ckpt_path.name} — seen avg success: {avg:.2%}")
        seen_rows.append(row)
        write_csv(seen_csv, seen_rows, seen_task_names)

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
            unseen_task_names: list[str] = [t["name"] for t in UNSEEN_TASKS]
            unseen_suite: str = ",".join(sorted(set(t["suite_name"] for t in UNSEEN_TASKS)))
            unseen_csv: str = os.path.join(args.output_dir, f"unseen_tasks_{args.mode}.csv")

            if not args.restart and os.path.exists(unseen_csv):
                unseen_completed, unseen_partial = _load_progress(unseen_csv)
                logger.info(
                    f"Resuming unseen evaluation: {len(unseen_completed)} checkpoint(s) already done."
                )
            else:
                unseen_completed, unseen_partial = {}, {}
                if args.restart:
                    logger.info("--restart: ignoring previous unseen-task results.")

            _init_csv(unseen_csv, unseen_task_names, force=args.restart)

            unseen_rows: list[dict] = [
                build_csv_row(p.name, unseen_completed[p.name], unseen_task_names, unseen_suite, init_states_str)[0]
                for p in ckpt_files
                if p.name in unseen_completed
            ]

            for ckpt_path in ckpt_files:
                if ckpt_path.name in unseen_completed:
                    logger.info(f"Skipping {ckpt_path.name} (already evaluated, unseen).")
                    continue

                logger.info(f"--- Evaluating (unseen): {ckpt_path.name} ---")
                model, model_config, stats = load_model(str(ckpt_path), args.env, device)

                def _on_unseen_task_done(
                    current_results: dict[str, dict[str, int]],
                    ckpt_name: str = ckpt_path.name,
                ) -> None:
                    partial_row, _ = build_csv_row(ckpt_name, current_results, unseen_task_names, unseen_suite, init_states_str)
                    write_csv(unseen_csv, unseen_rows + [partial_row], unseen_task_names, verbose=False)

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

                row, avg = build_csv_row(ckpt_path.name, task_results, unseen_task_names, unseen_suite, init_states_str)
                logger.info(f"{ckpt_path.name} — unseen avg success: {avg:.2%}")
                unseen_rows.append(row)
                write_csv(unseen_csv, unseen_rows, unseen_task_names)

    for env in seen_envs:
        env.close()



if __name__ == "__main__":
    main()
