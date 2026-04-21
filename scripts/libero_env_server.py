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
import subprocess
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zmq
import robosuite.utils.transform_utils as T

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


def _print_task_listing() -> None:
    """Print all available suites and their task indices to stdout."""
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    logger.info("Available suites and tasks:")
    for suite_key in LIBERO_CONFIGS:
        task_names: List[str] = libero_task_map.get(suite_key, [])
        logger.info(f"  [{suite_key}] ({len(task_names)} tasks)")
        for i, name in enumerate(task_names):
            logger.info(f"    [{i}] {name.replace('_', ' ')}")


def _load_task_suite(suite_name: str) -> Tuple[Any, List[Dict[str, Any]]]:
    """Load a LIBERO task suite and return (task_suite, tasks_list)."""
    from libero.libero import benchmark as _benchmark
    suite = _benchmark.get_benchmark_dict()[suite_name]()
    tasks: List[Dict[str, Any]] = []
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        tasks.append({
            "idx": i,
            "name": t.name,
            "description": get_task_description(t.name),
            "task": t,
        })
    return suite, tasks


def make_libero_env(task, cfg: Dict[str, Any]):
    """Create a LIBERO OffScreenRenderEnv for a specific task object.

    Always constructed with control_delta=True (delta mode) regardless of the
    control_delta argument.  handle_reset switches to absolute mode after the
    warmup by setting robot.controller.use_delta=False directly.  Creating with
    control_delta=False changes internal controller gain/bound parameters that
    cause instability when use_delta is later flipped at runtime; always using
    delta-mode construction (as eval_libero_abs_action does) avoids this.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file: str = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=cfg["image_size"],
        camera_widths=cfg["image_size"],
        control_delta=True,
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


def _patch_eef_obs(env, obs: dict) -> dict:
    """Replace robot0_eef_pos/quat with the gripper0_grip_site values.
    This makes obs["robot0_eef_quat"] match with controller.ee_ori_mat
    in terms of orientation.

    robosuite 1.4.1 reports robot0_eef_quat from a fingertip site whose
    orientation frame differs from the OSC_POSE controller's tracked body
    (robot0_right_hand) by a configuration-dependent rotation.  Sending
    the observable quaternion as an absolute orientation target therefore
    causes ~45° drift (see https://github.com/ARISE-Initiative/robosuite/issues/615).
    This is fixed in robosuite 1.5.0.  Until we upgrade, we patch the
    observation by reading gripper0_grip_site, which shares the same
    orientation frame as the controller.
    """
    site_id = env.env.sim.model.site_name2id("gripper0_grip_site")
    obs["robot0_eef_pos"] = env.env.sim.data.site_xpos[site_id].copy()
    obs["robot0_eef_quat"] = T.mat2quat(
        env.env.sim.data.site_xmat[site_id].reshape(3, 3)
    )
    return obs


def handle_reset(
    env,
    init_state: Optional[np.ndarray] = None,
    control_delta: bool = False,
) -> Dict[str, Any]:
    """Reset the environment and return the initial observation.

    If init_state is provided, applies it after the standard reset so the
    environment starts from a specific fixed initial state instead of a
    random one.

    The env is always constructed with control_delta=True to skip first few frames
    right after reset by inputting zero relative action, then set to the passed
    control_delta.
    """
    obs: dict = env.reset()

    if init_state is not None:
        obs = env.set_init_state(init_state)

    # Warmup: let objects settle. Env is constructed in delta mode so zeros = no movement.
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))

    # Switch to absolute mode when requested.
    if not control_delta:
        for robot in env.env.robots:
            robot.controller.use_delta = False

    return {"status": "ok", "obs": _patch_eef_obs(env, obs)}


def handle_step(env, action: np.ndarray) -> Dict[str, Any]:
    """Step the environment and return (obs, reward, done)."""
    obs, reward, done, info = env.step(action)
    return {
        "status": "ok",
        "obs": _patch_eef_obs(env, obs),
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
    suite_cache: Dict[str, Any],
    cfg: Dict[str, Any],
    control_delta: bool,
    image_key: str,
    record: bool,
    shutdown: threading.Event,
) -> Tuple[bool, List[dict], str]:
    """Run a single client session.

    The environment is created lazily when the client sends a "reset" with
    a task_idx, and recreated if the task_idx or suite_name changes.

    Returns (should_continue, obs_history, final_suite_name).
    """
    current_suite_name: str = cfg["env_name"]
    task_suite, tasks = suite_cache[current_suite_name]
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
                {"idx": t["idx"], "name": t["name"], "description": t["description"], "suite_name": current_suite_name}
                for t in tasks
            ]
            response = {"status": "ok", "tasks": task_info}

        elif cmd == "reset":
            suite_name: str = request.get("suite_name", current_suite_name)
            if suite_name != current_suite_name:
                if suite_name not in LIBERO_CONFIGS:
                    response = {
                        "status": "error",
                        "message": f"Unknown suite '{suite_name}'. "
                        f"Choose from: {list(LIBERO_CONFIGS.keys())}",
                    }
                    socket.send(pickle.dumps(response))
                    continue
                if suite_name not in suite_cache:
                    logger.info(f"Loading suite: {suite_name}")
                    suite_cache[suite_name] = _load_task_suite(suite_name)
                task_suite, tasks = suite_cache[suite_name]
                current_suite_name = suite_name
                cfg = LIBERO_CONFIGS[suite_name]
                image_key = cfg.get("image_key", "agentview_image")
                init_states_cache.clear()
                if env is not None:
                    close_env(env)
                    env = None
                current_task_idx = None
                logger.info(f"Switched to suite: {suite_name} ({len(tasks)} tasks)")

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
                    f"Loading task [{current_suite_name}:{task_idx}]: {tasks[task_idx]['description']}"
                )
                env = make_libero_env(tasks[task_idx]["task"], cfg)

                current_task_idx = task_idx

            response = handle_reset(env, init_state, control_delta=control_delta)
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
            return True, obs_history, current_suite_name

        elif cmd == "ping":
            response = {"status": "ok"}

        else:
            response = {"status": "error", "message": f"Unknown command: {cmd}"}

        socket.send(pickle.dumps(response))

    # Clean up on shutdown
    if env is not None:
        close_env(env)
    return False, obs_history, current_suite_name


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

    # Load initial task suite; additional suites are loaded on demand when a
    # client reset request includes a different suite_name.
    task_suite, tasks = _load_task_suite(cfg["env_name"])
    suite_cache: Dict[str, Any] = {cfg["env_name"]: (task_suite, tasks)}
    logger.info(f"Active suite: {cfg['env_name']} ({len(tasks)} tasks)")

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
            should_continue, obs_history, active_suite = run_session(
                socket,
                suite_cache,
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
            active_suite = env_key
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            try:
                socket.send(pickle.dumps({"status": "error", "message": str(e)}))
            except zmq.ZMQError:
                pass
            should_continue = True
            obs_history = []
            active_suite = env_key

        # Save video if recording was enabled and frames were collected
        if record and obs_history:
            save_video(obs_history, video_dir, active_suite, camera_keys)

        logger.info(f"Session {session_num}: ended")

        if not should_continue:
            break

    logger.info("Shutting down server")
    socket.close()
    context.term()


# ---------------------------------------------------------------------------
# Multi-server launcher
# ---------------------------------------------------------------------------

# ANSI color codes for distinguishing server output in the terminal
_SERVER_COLORS: List[str] = [
    "\033[36m",  # cyan
    "\033[33m",  # yellow
    "\033[35m",  # magenta
    "\033[32m",  # green
    "\033[34m",  # blue
    "\033[31m",  # red
    "\033[37m",  # white
    "\033[93m",  # bright yellow
]
_RESET: str = "\033[0m"


def _stream_server(idx: int, port: int, proc: subprocess.Popen) -> None:
    """Read a child server's stdout and reprint with a colored prefix."""
    color: str = _SERVER_COLORS[idx % len(_SERVER_COLORS)]
    prefix: str = f"{color}[S{idx} :{port}]{_RESET} "
    for line in proc.stdout:
        print(f"{prefix}{line}", end="", flush=True)


def _kill_port(port: int) -> None:
    """Kill any process currently bound to the given TCP port (best-effort)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid:
                subprocess.run(["kill", pid], check=False)
                logger.info(f"Killed stale process {pid} on port {port}")
    except FileNotFoundError:
        pass  # lsof not available


def run_multi_server(args: argparse.Namespace) -> None:
    """Spawn --num-servers child server processes on consecutive ports.

    Each child runs this same script with --num-servers 1 so they don't
    recurse. Their stdout is multiplexed into the current terminal with a
    color-coded prefix so you can watch all servers at once. Ctrl-C shuts
    everything down cleanly.
    """
    # Clear stale processes on the target ports before binding
    for i in range(args.num_servers):
        _kill_port(args.port + i)

    processes: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []

    for i in range(args.num_servers):
        port: int = args.port + i
        cmd: List[str] = [
            sys.executable, __file__,
            "--env", args.env,
            "--port", str(port),
            "--num-servers", "1",
            "--video-dir", args.video_dir,
            "--video-cameras", *args.video_cameras,
        ]
        if args.save_video:
            cmd.append("--save-video")
        if args.delta_actions:
            cmd.append("--delta-actions")

        env = {**os.environ, "PYTHONUNBUFFERED": "1", "LIBERO_WORKER": "1"}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        processes.append(proc)
        logger.info(f"Started server {i} on port {port} (pid {proc.pid})")

        t = threading.Thread(
            target=_stream_server, args=(i, port, proc), daemon=True
        )
        t.start()
        threads.append(t)

    def _shutdown(_sig: int, _frame) -> None:
        logger.info("Shutting down all servers...")
        for p in processes:
            p.terminate()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for proc in processes:
        proc.wait()
    logger.info("All servers stopped.")


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
    parser.add_argument(
        "--num-servers",
        type=int,
        default=1,
        metavar="N",
        help="Spawn N servers on consecutive ports starting at --port (default: 1). "
             "Output from all servers is multiplexed with a color-coded prefix.",
    )
    args = parser.parse_args()

    # Shorthand: --video-cameras both
    if args.video_cameras == ["both"]:
        args.video_cameras = ["agentview_image", "robot0_eye_in_hand_image"]

    import os as _os
    if not _os.environ.get("LIBERO_WORKER"):
        _print_task_listing()

    if args.num_servers > 1:
        run_multi_server(args)
    else:
        run_server(
            env_key=args.env,
            port=args.port,
            record=args.save_video,
            video_dir=args.video_dir,
            camera_keys=args.video_cameras,
            control_delta=args.delta_actions,
        )
