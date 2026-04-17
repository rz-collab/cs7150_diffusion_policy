# ---
# Generated: 2026-04-14 | claude-opus-4-6
# Prompt: ZMQ client that wraps a remote LIBERO environment server, providing
#         the same reset/step/render/close interface so inference code can use
#         it as a drop-in replacement for a local env.
# Modifications:
#   2026-04-14 | Prompt: Fix numpy version mismatch | Convert action to list before
#               pickling so numpy 2.x arrays don't reference numpy._core when
#               unpickled by the numpy 1.x server
#   2026-04-15 | Prompt: Task selection support | Added get_tasks() to query
#               available tasks from the server, and optional task_idx param
#               to reset() so the client can choose which task to load.
#   2026-04-17 | Prompt: Fixed initial state support | Added optional init_state_idx
#               param to reset() so the client can request a specific fixed initial
#               state (0–49) from the server's .init file for the current task.
# ---

"""
Remote environment client that communicates with a LIBERO ZMQ server.

Provides the same interface as a local environment (reset, step, render, close)
so that inference code can use it without knowing whether the env is local or
remote.
"""

import pickle
from typing import Optional

import numpy as np
import zmq


class RemoteEnv:
    """ZMQ REQ client that proxies env calls to a remote server."""

    def __init__(self, address: str = "tcp://localhost:5555", timeout_ms: int = 30000) -> None:
        self._address = address
        self._context: zmq.Context = zmq.Context()
        self._socket: zmq.Socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(address)
        # Verify the server is reachable
        self._send({"cmd": "ping"})

    def _send(self, request: dict) -> dict:
        """Send a request and return the response, raising on errors."""
        self._socket.send(pickle.dumps(request))
        raw: bytes = self._socket.recv()
        response: dict = pickle.loads(raw)
        if response.get("status") == "error":
            raise RuntimeError(
                f"Remote env error: {response.get('message', 'unknown')}"
            )
        return response

    def get_tasks(self) -> list[dict]:
        """Query the server for available tasks.

        Returns a list of dicts with keys: idx, name, description.
        """
        response = self._send({"cmd": "get_tasks"})
        return response["tasks"]

    def reset(self, task_idx: int | None = None, init_state_idx: int | None = None) -> dict:
        """Reset the remote environment and return the observation dict.

        Args:
            task_idx: If provided, load this task before resetting.
                      If None, reuse the current task (or default to 0).
            init_state_idx: If provided, load this fixed initial state (0–49)
                            from the task's .init file instead of a random one.
        """
        request: dict = {"cmd": "reset"}
        if task_idx is not None:
            request["task_idx"] = task_idx
        if init_state_idx is not None:
            request["init_state_idx"] = init_state_idx
        response = self._send(request)
        obs: dict = response["obs"]
        return obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Step the remote environment. Returns (obs, reward, done, info)."""
        # Convert to list to avoid numpy pickle incompatibility (2.x -> 1.x)
        response = self._send({"cmd": "step", "action": action.tolist()})
        obs: dict = response["obs"]
        reward: float = response["reward"]
        done: bool = response["done"]
        return obs, reward, done, {}

    def render(self) -> Optional[np.ndarray]:
        """Get the latest rendered frame from the remote environment."""
        response = self._send({"cmd": "render"})
        frame: Optional[np.ndarray] = response.get("frame")
        return frame

    def close(self) -> None:
        """Close the remote environment and clean up the socket."""
        try:
            self._send({"cmd": "close"})
        except (zmq.ZMQError, RuntimeError):
            pass
        self._socket.close()
        self._context.term()

    def __repr__(self) -> str:
        return f"RemoteEnv(address={self._address!r})"
