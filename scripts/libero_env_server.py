# ---
# Generated: 2026-04-14 | claude-opus-4-6
# Prompt: ZMQ server that wraps a LIBERO environment so the diffusion policy
#         (running in a separate conda env) can send actions and receive
#         observations over a socket.
# Modifications:
#   2026-04-14 | Prompt: Fix numpy version mismatch | Convert incoming action from
#               list to np.array since the client sends lists to avoid numpy 2.x/1.x
#               pickle incompatibility
#   2026-04-14 | Prompt: Fix render for OffScreenRenderEnv | OffScreenRenderEnv has
#               no render() method; return agentview_image from the last observation
#               instead
#   2026-04-14 | Prompt: Auto-restart, fix close, add video | Server now recreates
#               the environment after each client session instead of shutting down.
#               Fixed double-close crash by only closing env once per session. Added
#               --save-video flag to record agentview frames to mp4 each session.
#   2026-04-14 | Prompt: Fix video flip and resolution | Flip frames vertically
#               (MuJoCo origin is bottom-left) and upscale to 512x512 for viewing.
#               Only affects saved video, not model observations.
#   2026-04-14 | Prompt: Configurable video cameras | Added --video-cameras flag to
#               select which camera views to include in the saved video. Multiple
#               cameras are rendered side-by-side. Use --video-cameras both for
#               agentview + eye-in-hand.
#   2026-04-14 | Prompt: Make control_delta a changeable setting | Added control_delta
#               to LIBERO_CONFIGS and --absolute-actions CLI flag. Passes
#               control_delta through to ControlEnv so the OSC_POSE controller
#               can operate in either delta or absolute position mode.
#   2026-04-15 | Prompt: Client-driven task selection | Server loads the full
#               task suite on startup. Added "get_tasks" command returning
#               available tasks with descriptions. "reset" now accepts an
#               optional task_idx so the client can choose which task to run.
#               Env is created lazily on first reset and recreated when
#               task_idx changes.
#   2026-04-15 | Prompt: Add libero_10 env config | Added "libero_10" entry to
#               LIBERO_CONFIGS so the --env flag accepts libero_10 as a choice.
#   2026-04-15 | Prompt: Add all LIBERO task suites | Added libero_object,
#               libero_goal, libero_90, and libero_100 to LIBERO_CONFIGS. Set
#               control_delta=False on all suites (including libero_spatial) so
#               the controller uses absolute target poses.
#   2026-04-15 | Prompt: Change default to absolute actions | Flipped
#               --absolute-actions to --delta-actions so the default is absolute
#               position mode, matching the config values in LIBERO_CONFIGS.
#   2026-04-17 | Prompt: Fixed initial state support | Added optional init_state_idx
#               to reset command so client can load a specific fixed initial state
#               (0–49) from the task's .init file via task_suite.get_task_init_states().
#               Updated handle_reset to call env.set_init_state() after env.reset(),
#               added init_states_cache in run_session to avoid redundant disk reads,
#               added task_suite param to run_session and threaded it through run_server.
#               Fixed: get_task_init_states returns np.ndarray in this LIBERO version,
#               not a torch.Tensor; use hasattr guard instead of unconditional .numpy().
# ---

"""
LIBERO environment ZMQ server.

Run this script inside the `libero` conda environment. It exposes the LIBERO
simulation environment over a ZMQ REP socket so that the diffusion policy
inference code (running in the `diff_policy` env) can interact with it.

The server persists across client sessions — when a client disconnects (sends
"close"), the environment is recreated and the server waits for the next client.
Use Ctrl-C to shut down the server.

Usage:
    conda activate libero
    python scripts/libero_env_server.py --env libero_spatial --port 5555
    python scripts/libero_env_server.py --env libero_spatial --save-video
"""

import argparse
import logging
import os
import pickle
import re
import signal
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment helpers (LIBERO-specific imports happen here, inside the libero env)
# ---------------------------------------------------------------------------

# Minimal env config duplicated here so the server is self-contained and does
# not need to import from the diffusion_policy package (which lives in the
# other conda env).
LIBERO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "libero_spatial": {
        "env_name": "libero_spatial",
        "image_size": 128,
        "image_key": "agentview_image",
        "control_delta": False,
    },
    "libero_object": {
        "env_name": "libero_object",
        "image_size": 128,
        "image_key": "agentview_image",
        "control_delta": False,
    },
    "libero_goal": {
        "env_name": "libero_goal",
        "image_size": 128,
        "image_key": "agentview_image",
        "control_delta": False,
    },
    "libero_10": {
        "env_name": "libero_10",
        "image_size": 128,
        "image_key": "agentview_image",
        "control_delta": False,
    },
    "libero_90": {
        "env_name": "libero_90",
        "image_size": 128,
        "image_key": "agentview_image",
        "control_delta": False,
    },
    "libero_100": {
        "env_name": "libero_100",
        "image_size": 128,
        "image_key": "agentview_image",
        "control_delta": False,
    },
}


def get_task_description(task_name: str) -> str:
    """Extract natural language description from a LIBERO task name."""
    name = re.sub(r"_demo$", "", task_name)
    name = re.sub(r"^[A-Z_]+SCENE\d+_", "", name)
    return name.replace("_", " ")


def make_libero_env(task, cfg: Dict[str, Any], control_delta: bool = True):
    """Create a LIBERO OffScreenRenderEnv for a specific task object."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file: str = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=cfg["image_size"],
        camera_widths=cfg["image_size"],
        control_delta=control_delta,
    )


def close_env(env) -> None:
    """Safely close a LIBERO environment, handling the missing self.env case."""
    if hasattr(env, "env"):
        env.close()


def save_video(
    obs_history: List[dict],
    video_dir: str,
    env_key: str,
    camera_keys: List[str],
    output_size: int = 512,
) -> None:
    """Save collected observations to an mp4, rendering the selected cameras.

    If multiple camera_keys are given, they are placed side-by-side in each
    frame. Frames are flipped vertically (MuJoCo renders origin at bottom-left)
    and upscaled for easier viewing. This only affects the saved video — the
    observations sent to the model are untouched.
    """
    if not obs_history:
        logger.info("No frames to save")
        return

    import cv2
    import imageio

    processed: List[np.ndarray] = []
    for obs in obs_history:
        panels: List[np.ndarray] = []
        for key in camera_keys:
            if key not in obs:
                continue
            img: np.ndarray = np.flipud(obs[key])
            img = cv2.resize(
                img,
                (output_size, output_size),
                interpolation=cv2.INTER_NEAREST,
            )
            panels.append(img)
        if panels:
            processed.append(np.concatenate(panels, axis=1))

    if not processed:
        logger.info("No camera frames found in observations")
        return

    os.makedirs(video_dir, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path: str = os.path.join(video_dir, f"{env_key}_{timestamp}.mp4")
    h, w = processed[0].shape[:2]
    imageio.mimwrite(path, processed, fps=20)
    logger.info(f"Saved video ({len(processed)} frames, {w}x{h}) to {path}")


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


def handle_reset(env, init_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Reset the environment and return the initial observation.

    If init_state is provided, applies it after the standard reset so the
    environment starts from a specific fixed initial state instead of a
    random one.
    """
    obs: dict = env.reset()

    if init_state is not None:
        obs = env.set_init_state(init_state)

    # When env resets (even if we set init state),
    # objects are dropped unnaturally from some height. Skip these initial frames
    # by doing nothing.
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))

    return {"status": "ok", "obs": obs}


def handle_step(env, action: np.ndarray) -> Dict[str, Any]:
    """Step the environment and return (obs, reward, done)."""
    obs, reward, done, info = env.step(action)
    return {
        "status": "ok",
        "obs": obs,
        "reward": float(reward),
        "done": bool(done),
    }


def handle_render(last_obs: Optional[dict], image_key: str) -> Dict[str, Any]:
    """Return the rendered frame from the last observation.

    LIBERO's OffScreenRenderEnv has no render() method — the image is
    already included in the observation dict.
    """
    frame: Optional[np.ndarray] = None
    if last_obs is not None and image_key in last_obs:
        frame = last_obs[image_key]
    return {"status": "ok", "frame": frame}


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------


def run_session(
    socket: zmq.Socket,
    tasks: List[Dict[str, Any]],
    task_suite,
    cfg: Dict[str, Any],
    control_delta: bool,
    image_key: str,
    record: bool,
    shutdown: threading.Event,
) -> Tuple[bool, List[dict]]:
    """Run a single client session.

    The environment is created lazily when the client sends a "reset" with
    a task_idx, and recreated if the task_idx changes.

    Returns (should_continue, obs_history).
    """
    env = None
    current_task_idx: Optional[int] = None
    last_obs: Optional[dict] = None
    obs_history: List[dict] = []
    init_states_cache: Dict[int, Any] = {}

    while not shutdown.is_set():
        if not socket.poll(timeout=1000):
            continue

        raw: bytes = socket.recv()
        request: Dict[str, Any] = pickle.loads(raw)
        cmd: str = request.get("cmd", "")

        if cmd == "get_tasks":
            task_info = [
                {"idx": t["idx"], "name": t["name"], "description": t["description"]}
                for t in tasks
            ]
            response = {"status": "ok", "tasks": task_info}

        elif cmd == "reset":
            task_idx: int = request.get(
                "task_idx",
                current_task_idx if current_task_idx is not None else 0,
            )
            if task_idx < 0 or task_idx >= len(tasks):
                response = {
                    "status": "error",
                    "message": f"Invalid task_idx {task_idx}. "
                    f"Must be 0–{len(tasks) - 1}.",
                }
                socket.send(pickle.dumps(response))
                continue

            init_state_idx: Optional[int] = request.get("init_state_idx")

            # Load and cache init states if a specific state is requested
            init_state: Optional[np.ndarray] = None
            if init_state_idx is not None:
                if task_idx not in init_states_cache:
                    init_states_cache[task_idx] = task_suite.get_task_init_states(
                        task_idx
                    )
                task_init_states = init_states_cache[task_idx]
                num_states: int = len(task_init_states)
                if init_state_idx < 0 or init_state_idx >= num_states:
                    response = {
                        "status": "error",
                        "message": f"Invalid init_state_idx {init_state_idx}. "
                        f"Must be 0–{num_states - 1}.",
                    }
                    socket.send(pickle.dumps(response))
                    continue
                raw_state = task_init_states[init_state_idx]
                # get_task_init_states may return a torch.Tensor or np.ndarray
                init_state = (
                    raw_state.numpy()
                    if hasattr(raw_state, "numpy")
                    else np.array(raw_state)
                )

            # Create or recreate the env when the task changes
            if env is None or task_idx != current_task_idx:
                if env is not None:
                    close_env(env)
                logger.info(
                    f"Loading task {task_idx}: {tasks[task_idx]['description']}"
                )
                env = make_libero_env(tasks[task_idx]["task"], cfg, control_delta)
                current_task_idx = task_idx

            response = handle_reset(env, init_state)
            last_obs = response.get("obs")
            if record and last_obs is not None:
                obs_history.append(last_obs)

        elif cmd == "step":
            if env is None:
                response = {
                    "status": "error",
                    "message": "No env loaded. Send reset first.",
                }
            else:
                action = np.array(request["action"], dtype=np.float32)
                response = handle_step(env, action)
                last_obs = response.get("obs")
                if record and last_obs is not None:
                    obs_history.append(last_obs)

        elif cmd == "render":
            response = handle_render(last_obs, image_key)

        elif cmd == "close":
            if env is not None:
                close_env(env)
            socket.send(pickle.dumps({"status": "ok"}))
            return True, obs_history

        elif cmd == "ping":
            response = {"status": "ok"}

        else:
            response = {"status": "error", "message": f"Unknown command: {cmd}"}

        socket.send(pickle.dumps(response))

    # Clean up on shutdown
    if env is not None:
        close_env(env)
    return False, obs_history


def run_server(
    env_key: str,
    port: int,
    record: bool,
    video_dir: str,
    camera_keys: List[str],
    control_delta: bool = True,
) -> None:
    """Start the ZMQ REP server and serve environment sessions in a loop."""
    cfg = LIBERO_CONFIGS[env_key]
    image_key: str = cfg.get("image_key", "agentview_image")
    action_mode: str = "delta" if control_delta else "absolute"
    logger.info(f"Action mode: {action_mode}")

    # Load task suite once at startup
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[cfg["env_name"]]()
    num_tasks: int = task_suite.n_tasks
    tasks: List[Dict[str, Any]] = []
    for i in range(num_tasks):
        task = task_suite.get_task(i)
        tasks.append(
            {
                "idx": i,
                "name": task.name,
                "description": get_task_description(task.name),
                "task": task,
            }
        )
    logger.info(f"Loaded {num_tasks} tasks from {cfg['env_name']}:")
    for t in tasks:
        logger.info(f"  [{t['idx']}] {t['description']}")

    # ZMQ setup
    context: zmq.Context = zmq.Context()
    socket: zmq.Socket = context.socket(zmq.REP)
    address: str = f"tcp://*:{port}"
    socket.bind(address)
    logger.info(f"Server listening on {address}")

    # Graceful shutdown on Ctrl-C
    shutdown = threading.Event()

    def _signal_handler(sig: int, frame) -> None:
        logger.info("Shutdown signal received")
        shutdown.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    session_num: int = 0

    while not shutdown.is_set():
        session_num += 1
        logger.info(f"Session {session_num}: waiting for client")

        try:
            should_continue, obs_history = run_session(
                socket,
                tasks,
                task_suite,
                cfg,
                control_delta,
                image_key,
                record,
                shutdown,
            )
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                break
            logger.error(f"ZMQ error: {e}")
            should_continue = False
            obs_history = []
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            try:
                socket.send(pickle.dumps({"status": "error", "message": str(e)}))
            except zmq.ZMQError:
                pass
            should_continue = True
            obs_history = []

        # Save video if recording was enabled and frames were collected
        if record and obs_history:
            save_video(obs_history, video_dir, env_key, camera_keys)

        logger.info(f"Session {session_num}: ended")

        if not should_continue:
            break

    logger.info("Shutting down server")
    socket.close()
    context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO environment ZMQ server")
    parser.add_argument(
        "--env",
        type=str,
        default="libero_spatial",
        choices=list(LIBERO_CONFIGS.keys()),
        help="LIBERO environment config key (default: libero_spatial)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ port to bind to (default: 5555)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save camera frames to an mp4 video each session",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="runs/libero_videos",
        help="Directory to save videos (default: runs/libero_videos)",
    )
    parser.add_argument(
        "--video-cameras",
        type=str,
        nargs="+",
        default=["agentview_image"],
        help="Camera keys to include in video, side-by-side "
        "(default: agentview_image). "
        "Use 'both' for agentview_image + robot0_eye_in_hand_image.",
    )
    parser.add_argument(
        "--delta-actions",
        action="store_true",
        help="Use delta actions instead of absolute target poses "
        "(sets control_delta=True on the OSC_POSE controller)",
    )
    args = parser.parse_args()

    # Shorthand: --video-cameras both
    if args.video_cameras == ["both"]:
        args.video_cameras = ["agentview_image", "robot0_eye_in_hand_image"]

    run_server(
        env_key=args.env,
        port=args.port,
        record=args.save_video,
        video_dir=args.video_dir,
        camera_keys=args.video_cameras,
        control_delta=args.delta_actions,
    )
