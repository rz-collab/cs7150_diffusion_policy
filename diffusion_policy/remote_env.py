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
#   2026-04-18 | Prompt: Separate ping and op timeouts | Split timeout_ms into
#               ping_timeout_ms (default 60s) used only for the startup ping and
#               op_timeout_ms (default 600s) reapplied to the socket afterwards.
#               Task switching forces the server to rebuild a LIBERO env which
#               exceeded the previous 60s single-timeout budget and tripped
#               zmq.Again mid-run.
#   2026-04-19 | Prompt: Add suite_name to reset so one server handles all suites |
#               Added suite_name param to reset(); when provided, included in the
#               request so the server can switch to a different task suite before
#               resetting (requires server-side suite switching support).
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

    def __init__(
        self,
        address: str = "tcp://localhost:5555",
        ping_timeout_ms: int = 60000,
        op_timeout_ms: int = 600000,
    ) -> None:
        self._address = address
        self._context: zmq.Context = zmq.Context()
        self._socket: zmq.Socket = self._context.socket(zmq.REQ)
        # Short timeout for ping so a missing server fails fast at startup.
        self._socket.setsockopt(zmq.RCVTIMEO, ping_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, ping_timeout_ms)
        self._socket.connect(address)
        self._send({"cmd": "ping"})
        # Relax timeouts for real operations: rebuilding a LIBERO env on task
        # switch can easily take longer than a minute when several servers are
        # loading in parallel.
        self._socket.setsockopt(zmq.RCVTIMEO, op_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, op_timeout_ms)

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

    def reset(self, task_idx: int | None = None, init_state_idx: int | None = None, suite_name: str | None = None) -> dict:
        """Reset the remote environment and return the observation dict.

        Args:
            task_idx: If provided, load this task before resetting.
                      If None, reuse the current task (or default to 0).
            init_state_idx: If provided, load this fixed initial state (0–49)
                            from the task's .init file instead of a random one.
            suite_name: If provided and different from the current suite on the
                        server, the server switches to that suite before resetting.
        """
        request: dict = {"cmd": "reset"}
        if task_idx is not None:
            request["task_idx"] = task_idx
        if init_state_idx is not None:
            request["init_state_idx"] = init_state_idx
        if suite_name is not None:
            request["suite_name"] = suite_name
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
