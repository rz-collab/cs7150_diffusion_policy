# Claude Code Ledger

This file tracks all changes made to this codebase with Claude Code assistance.
Each section corresponds to a source file, showing when it was generated and all
subsequent modifications with their prompts and descriptions.

---

## `diffusion_policy/env_config.py`

**Generated:** 2026-04-06 03:00 UTC | `claude-opus-4-6`  
**Prompt:** Create a shared environment config system so inference (and later training) can switch between PushT, LIBERO, and other envs via a single flag.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Add ZMQ socket support for LIBERO | Added `zmq_address` field to LIBERO configs so inference connects to the remote env server instead of importing libero directly |
| 2026-04-14 | Make control_delta a changeable setting | Added `control_delta` field to LIBERO configs (default True) so users can switch between delta and absolute position action modes |
| 2026-04-15 | Centralize task_descriptions_path in env config | Added `task_descriptions_path` field to each env config so train and inference scripts read the path from config instead of hardcoding it |
| 2026-04-15 | Switch LIBERO to absolute actions | Changed `control_delta` from True to False so env config matches absolute-position mode used by all LIBERO task suites |
| 2026-04-19 | dataset_path is now a base directory; full path is dataset_path/train_task_suite | Changed libero `dataset_path` from `["data/libero_abs"]` (a list with the suite baked in) to the bare string `"data/libero_abs"` so callers join it with `train_task_suite` at runtime. This lets you point at different suites without editing the path field directly |

---

## `diffusion_policy/model/encoder_base.py`

**Generated:** 2026-04-17 | `claude-sonnet-4-6`  
**Prompt:** Shared abstract base class for visual and language encoders, extracted from VisualEncoder in visual_encoder.py so both encoder families can inherit from a single common interface.

*(No subsequent modifications)*

---

## `diffusion_policy/model/visual_encoder.py`

**Generated:** 2026-04-06 | `claude-opus-4-6`  
**Prompt:** ResNet-18 visual encoder, copied from Diffusion Policy Colab's Notebook.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Add CLIP vision encoder option | Added `CLIPEncoder` that wraps CLIP ViT-B/32 vision model. Resizes and normalizes inputs to CLIP's expected format, outputs 512-dim features matching the ResNet-18 interface |
| 2026-04-15 | Add DINOv2 vision encoder | Added `dinov2-small/base/large` entries to `PRETRAINED_VISION_MODELS` and `DINOv2VisualEncoder` class that wraps HuggingFace `Dinov2Model`, using CLS token output with ImageNet normalization and auto-resize |
| 2026-04-15 | Add vision projection layer | Added optional `proj_dim` parameter to `PretrainedVisualEncoder` and `DINOv2VisualEncoder`. When set, a trainable 2-layer MLP (Linear→Mish→Linear) projects encoder output to `proj_dim`, updating `output_dim` accordingly. None (default) preserves original behavior with no projection |
| 2026-04-17 | Option B freeze_backbone | Add `freeze_backbone()` method to `DINOv2VisualEncoder` and `PretrainedVisualEncoder` so callers can freeze only the pretrained backbone weights while leaving the trainable projection MLP (`self.proj`) unfrozen |
| 2026-04-17 | Abstract base class for vision encoders | Added `Encoder` ABC with abstract `freeze_backbone()` and updated `DINOv2VisualEncoder` and `PretrainedVisualEncoder` to inherit from it. `get_visual_encoder()` still returned plain `nn.Module` |
| 2026-04-17 | Update ResNet for Encoder ABC | Wrapped ResNet-18 in `ResNetVisualEncoder(Encoder)` with `output_dim=512`. `freeze_backbone()` raises `NotImplementedError` since ResNet-18 has no separate projection layer. Updated `get_visual_encoder()` to return `ResNetVisualEncoder` |
| 2026-04-17 | Unify encoder base class | Moved `Encoder` ABC into `encoder_base.py` as `Encoder` so `LanguageEncoder` can share it. Removed `Encoder` class; all subclasses now inherit `Encoder` |
| 2026-04-17 | Test all encoders | Fixed `PretrainedVisualEncoder` to use `nn.Identity()` when the model has no `visual_projection` (e.g. SigLIP), since SigLIP's `pooler_output` is already the final embedding with no separate projection head |
| 2026-04-17 | Unify encoder base class | Renamed `Encoder` so `LanguageEncoder` can also inherit from it. Updated all subclass declarations and exports accordingly |

---

## `diffusion_policy/model/language_encoder.py`

**Generated:** 2026-04-14 | `claude-opus-4-6`  
**Prompt:** Create a language encoder module supporting both a pure text encoder and a CLIP encoder, with configurable projection dimension and freeze setting, for use as conditioning in the diffusion policy.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Make pretrained model configurable | Replaced hardcoded CLIP model with a `pretrained_model` param that looks up the model from `PRETRAINED_VISION_MODELS` (shared with `visual_encoder.py`). Supports CLIP and SigLIP family text encoders alongside the standalone "text" backend |
| 2026-04-17 | Unify encoder base class | `LanguageEncoder` now inherits from `Encoder` (`encoder_base.py`). Added `freeze_backbone()` that freezes `self.encoder` only, leaving `self.proj` trainable. `__init__` calls `freeze_backbone()` instead of inlining the parameter loop |

---

## `diffusion_policy/model/diffusion_policy.py`

**Generated:** 2026-04-06 | `claude-opus-4-6`  
**Prompt:** Wrapper class that wraps all components (diffusion denoiser, diff step encoder, visual encoder) into a single pytorch model.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Add exchangeable language conditioning | Added optional language encoder ("clip" or "text") with configurable projection dim and freeze flag. Model stores its constructor config as `model_config` for checkpoint serialization |
| 2026-04-14 | Add CLIP vision encoder option | Added `use_clip_vision` flag to swap ResNet-18 for CLIP ViT-B/32 vision encoder, and `freeze_visual_encoder` flag to control whether it trains |
| 2026-04-14 | Consolidate encoder settings into single string | Replaced separate `use_clip_vision` / `lang_encoder_type` / freeze flags with a single `encoder_type` string ("resnet_only", "clip_text", "clip_both", "resnet_and_text") and one `freeze_encoders` bool |
| 2026-04-15 | Add DINOv2 encoder support | Added "dino_only", "dino_clip_text", and "dino_text" encoder types. DINOv2 vision is paired with no language, a pretrained CLIP/SigLIP text encoder, or a standalone text encoder respectively |
| 2026-04-15 | Replace encoder_type with vision/text encoder keys | Replaced `encoder_type`, `pretrained_model`, `text_pretrained_model` with two simple params: `vision_encoder` (model key or None for ResNet-18) and `text_encoder` (model key, "text", or None for no language). When both point to the same clip/siglip model, weights are shared. Old checkpoint `model_config`s are converted automatically |
| 2026-04-15 | Add vision projection layer | Added `vision_proj_dim` parameter (default 512) that adds a trainable projection MLP to pretrained vision encoders, matching ResNet-18 output dim. Does not apply to ResNet-18 itself. Legacy checkpoint configs default to None (no projection) for backwards compatibility |
| 2026-04-17 | Option B freeze_backbone | Projection layer should not freeze with the backbone. Replaced the blanket `parameters()` loop with a call to `visual_encoder.freeze_backbone()` for pretrained encoders (which only freezes the backbone, not `self.proj`). Falls back to freezing all parameters for ResNet-18, which has no projection layer |
| 2026-04-17 | Update ResNet for VisualEncoder ABC | All encoders now subclass `VisualEncoder`. Updated `self.visual_encoder` type annotation to `VisualEncoder`, removed `hasattr` fallback (ResNet raises `NotImplementedError`), and replaced hardcoded ResNet-18 `output_dim` with `self.visual_encoder.output_dim` |
| 2026-04-18 | Fix AttributeError on SiglipModel.visual_projection | Used `getattr(..., None)` when passing `visual_projection` in the shared-weights branch so SigLIP (which has no projection layer) falls through to the `nn.Identity()` fallback in `PretrainedVisualEncoder` |

---

## `diffusion_policy/dataset/pusht.py`

**Generated:** 2025-01-01 | `claude-opus-4-6`  
**Prompt:** `PushTImageDataset` and helper functions for loading pusht zarr data.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Add text descriptions for diffusion model conditioning | Added `descriptions_path` param to `PushTDataset`, loads JSON descriptions and returns a random one per sample in `__getitem__` |
| 2026-04-14 | Accept descriptions list directly | Changed from `descriptions_path` to a `descriptions` list param so the caller resolves the task from `task_descriptions.json` |
| 2026-04-14 | Per-sample language dropout | Moved language dropout from training loop into `__getitem__`. With `lang_dropout_prob`, individual samples return `""` instead of a real description, letting the model learn an unconditional embedding per-sample |

---

## `diffusion_policy/dataset/libero.py`

**Generated:** 2026-04-06 | `claude-opus-4-6`  
**Prompt:** LIBERO dataset loading with single-task wrappers and multi-task concat.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-15 | Add description support matching PushT dataset | Added `descriptions` and `lang_dropout_prob` params to `LiberoSingleTaskDataset` and `get_libero_dataset`. `__getitem__` now returns a "description" key with random sampling from per-task description paraphrases and per-sample language dropout. Falls back to extracted language when no descriptions are provided |

---

## `diffusion_policy/remote_env.py`

**Generated:** 2026-04-14 | `claude-opus-4-6`  
**Prompt:** ZMQ client that wraps a remote LIBERO environment server, providing the same reset/step/render/close interface so inference code can use it as a drop-in replacement for a local env.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Fix numpy version mismatch | Convert action to list before pickling so numpy 2.x arrays don't reference `numpy._core` when unpickled by the numpy 1.x server |
| 2026-04-15 | Task selection support | Added `get_tasks()` to query available tasks from the server, and optional `task_idx` param to `reset()` so the client can choose which task to load |
| 2026-04-17 | Fixed initial state support | Added optional `init_state_idx` param to `reset()` so the client can request a specific fixed initial state (0–49) from the server's `.init` file for the current task |
| 2026-04-18 | Separate ping and op timeouts | Split `timeout_ms` into `ping_timeout_ms` (default 60s) used only for the startup ping and `op_timeout_ms` (default 600s) reapplied to the socket afterwards. Task switching forces the server to rebuild a LIBERO env which exceeded the previous 60s single-timeout budget and tripped `zmq.Again` mid-run |
| 2026-04-19 | Add suite_name to reset so one server handles all suites | Added `suite_name` param to `reset()`; when provided, included in the request so the server can switch to a different task suite before resetting (requires server-side suite switching support) |

---

## `scripts/train.py`

**Generated:** 2026-04-06 | `claude-opus-4-6`  
**Prompt:** Training loop for diffusion policy with PushT environment.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Save model config in checkpoint and support language conditioning | Checkpoint now saves `model_config` alongside `state_dict` so the model can be reconstructed at inference. Added `LANG_ENCODER_TYPE` / `LANG_PROJ_DIM` / `FREEZE_LANG_ENCODER` settings. Optimizer filters out frozen parameters. `TASK_DESCRIPTION` passed to forward when language encoder is active |
| 2026-04-14 | Per-task descriptions from JSON with dropout | Descriptions loaded from `data/task_descriptions.json` keyed by `TASK_KEY` (and optional `TASK_SUBTASK` for LIBERO). Passed to dataset which returns a random description per sample. `LANG_DROPOUT_PROB` drops language conditioning for some batches |
| 2026-04-14 | Move lang dropout to dataset | Removed batch-level language dropout from training loop; per-sample dropout is now handled in `PushTDataset`. Removed unused `random` import |
| 2026-04-14 | Fix text encoder loading for resnet_only | `task_description` is now only extracted from the batch and passed to forward when the model has a language encoder, preventing a `KeyError` and avoiding unnecessary text processing for `resnet_only` encoder type |
| 2026-04-15 | Add description support for LIBERO | Refactored description loading to support both flat lists (PushT) and per-task dicts (LIBERO). Passes `task_descriptions_by_task` and `LANG_DROPOUT_PROB` to `get_libero_dataset` so LIBERO samples return a "description" key with random sampling and dropout |
| 2026-04-15 | Read task_descriptions_path from env config | Replaced hardcoded `TASK_DESCRIPTIONS_PATH` and `TASK_KEY` with `cfg["task_descriptions_path"]` and `cfg["task_descriptions_key"]` so the path is centralized in `env_config.py` |
| 2026-04-15 | tqdm-safe logging and description debug log | Added `TqdmLoggingHandler` so `logger.info` doesn't break progress bars. Log sample descriptions from the first batch to verify they reach the model |
| 2026-04-15 | Save data_stats in checkpoint | Saved a numpy copy of LIBERO normalization stats (`data_stats_np`) into the checkpoint so inference can unnormalize actions without scanning HDF5 files |
| 2026-04-15 | Replace encoder_type with vision/text encoder | Replaced `ENCODER_TYPE`, `PRETRAINED_MODEL`, `TEXT_PRETRAINED_MODEL` with `VISION_ENCODER` and `TEXT_ENCODER`. Updated `DiffusionPolicy` call and description-loading guard to use new params |
| 2026-04-19 | dataset_path is now a base dir; join with train_task_suite | Changed LIBERO data loading to compute the HDF5 folder as `os.path.join(cfg["dataset_path"], cfg["train_task_suite"])` so the full path is derived at runtime instead of hardcoded in config |

---

## `scripts/inference.py`

**Generated:** 2026-04-06 | `claude-opus-4-6`  
**Prompt:** Inference module for the diffusion policy PushT environment, including model loading, DDPM denoising loop, observation buffering, and environment execution.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-06 | Refactor to use shared env_config | Refactored so environments are swappable via `--env` flag (e.g. `--env pusht`, `--env libero_spatial`) |
| 2026-04-07 | Add type annotations | Added type annotations to all function parameters, return types, and non-obvious variable declarations |
| 2026-04-07 | Fix env visual rendering | Updated `make_env` to pass `obs_type` from config to `gym.make()` so observations include pixels and `agent_pos` as a dict |
| 2026-04-07 | Add visual display during inference | Added standalone pygame display window for visual rendering while keeping `render_mode=rgb_array` for correct observation capture. Renders env frames to a 512×512 window each step |
| 2026-04-08 | Fix PushT observation handling | Fixed inference to handle both flat observation arrays and dict observations, with fallback to `env.render()` for images |
| 2026-04-08 | Hardcode render_mode and obs_type into make_env | Moved `render_mode="rgb_array"` and `obs_type="pixels_agent_pos"` from env config into `make_env` since they are fixed inference requirements |
| 2026-04-08 | Remove dead extract functions | Removed `extract_image` and `extract_state_from_obs` since obs is always a dict now. Replaced usages with direct dict access and `extract_state` |
| 2026-04-14 | Use ZMQ sockets for LIBERO | Replaced direct LIBERO imports in `make_env` with `RemoteEnv` ZMQ client so LIBERO runs in its own conda env via a server, connected over a socket |
| 2026-04-14 | Support language-conditioned checkpoints | Checkpoint loading now reads `model_config` to reconstruct the model (including language encoder settings). Supports both new and legacy checkpoint formats. Task description passed to forward during denoising loop |
| 2026-04-14 | Random fallback description | Changed fallback task description selection from first entry to `random.choice`. Added log line showing which description is being used |
| 2026-04-15 | Add LIBERO inference support | Branched stats loading (HDF5 via `compute_stats_from_hdf5` for LIBERO, zarr `PushTDataset` for PushT), observation preprocessing (per-key state normalization and image /255 for LIBERO), noise tensor dims from `model_config`, and action denormalization (10D rot-6d model output to 7D axis-angle env actions via `denormalize_actions_libero`) |
| 2026-04-15 | Fix LIBERO model loading | Override `model_config` `action_dim` (7→10) and `state_obs_dim` to match the 6D rotation representation training uses. Reordered model loading before stats loading so shape errors surface before the slow HDF5 scan |
| 2026-04-15 | Make display and video optional | Replaced always-on pygame window and automatic video save with `--display` and `--save-video` flags. Neither runs by default; pygame and imageio are now conditional imports |
| 2026-04-15 | Client-driven task selection | For LIBERO, queries available tasks from the server via `get_tasks` and selects one (random or via `--task-idx`). Task description auto-populated from the server for language conditioning |
| 2026-04-15 | Read task_descriptions_path from env config | Replaced hardcoded `TASK_DESCRIPTIONS_PATH` with `cfg["task_descriptions_path"]` and `cfg["task_descriptions_key"]` so the path is centralized in `env_config.py` |
| 2026-04-15 | Load data_stats from checkpoint | LIBERO normalization stats are now loaded from the checkpoint when available, removing the HDF5 dependency at inference time. Falls back to computing from HDF5 files for older checkpoints |
| 2026-04-15 | Support new vision/text_encoder config | Import and apply `_convert_legacy_model_config` to translate old `encoder_type` checkpoint configs to the new `vision_encoder`/`text_encoder` params |
| 2026-04-15 | Fix weights_only load error | Changed `torch.load` to `weights_only=False` because the checkpoint contains numpy arrays (via `numpy._core.multiarray._reconstruct`) which are rejected by PyTorch 2.6+'s default `weights_only=True` safe-unpickling |
| 2026-04-19 | Add --suite flag for LIBERO task suite selection | Added `suite` param to `run_inference()` and `--suite` CLI arg. When specified, a preliminary `reset(suite_name=suite)` switches the server's active suite before `get_tasks()` so the returned task list matches the requested suite. `suite_name` is also forwarded to the final `reset()` |

---

## `scripts/libero_env_server.py`

**Generated:** 2026-04-14 | `claude-opus-4-6`  
**Prompt:** ZMQ server that wraps a LIBERO environment so the diffusion policy (running in a separate conda env) can send actions and receive observations over a socket.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Fix numpy version mismatch | Convert incoming action from list to `np.array` since the client sends lists to avoid numpy 2.x/1.x pickle incompatibility |
| 2026-04-14 | Fix render for OffScreenRenderEnv | `OffScreenRenderEnv` has no `render()` method; return `agentview_image` from the last observation instead |
| 2026-04-14 | Auto-restart, fix close, add video | Server now recreates the environment after each client session instead of shutting down. Fixed double-close crash by only closing env once per session. Added `--save-video` flag to record agentview frames to mp4 each session |
| 2026-04-14 | Fix video flip and resolution | Flip frames vertically (MuJoCo origin is bottom-left) and upscale to 512×512 for viewing. Only affects saved video, not model observations |
| 2026-04-14 | Configurable video cameras | Added `--video-cameras` flag to select which camera views to include in the saved video. Multiple cameras are rendered side-by-side. Use `--video-cameras both` for agentview + eye-in-hand |
| 2026-04-14 | Make control_delta a changeable setting | Added `control_delta` to `LIBERO_CONFIGS` and `--absolute-actions` CLI flag. Passes `control_delta` through to `ControlEnv` so the OSC_POSE controller can operate in either delta or absolute position mode |
| 2026-04-15 | Client-driven task selection | Server loads the full task suite on startup. Added `"get_tasks"` command returning available tasks with descriptions. `"reset"` now accepts an optional `task_idx` so the client can choose which task to run. Env is created lazily on first reset and recreated when `task_idx` changes |
| 2026-04-15 | Add libero_10 env config | Added `"libero_10"` entry to `LIBERO_CONFIGS` so the `--env` flag accepts `libero_10` as a choice |
| 2026-04-15 | Add all LIBERO task suites | Added `libero_object`, `libero_goal`, `libero_90`, and `libero_100` to `LIBERO_CONFIGS`. Set `control_delta=False` on all suites (including `libero_spatial`) so the controller uses absolute target poses |
| 2026-04-15 | Change default to absolute actions | Flipped `--absolute-actions` to `--delta-actions` so the default is absolute position mode, matching the config values in `LIBERO_CONFIGS` |
| 2026-04-17 | Fixed initial state support | Added optional `init_state_idx` to reset command so client can load a specific fixed initial state (0–49) from the task's `.init` file via `task_suite.get_task_init_states()`. Updated `handle_reset` to call `env.set_init_state()` after `env.reset()`, added `init_states_cache` in `run_session` to avoid redundant disk reads, added `task_suite` param to `run_session` and threaded it through `run_server`. Fixed: `get_task_init_states` returns `np.ndarray` in this LIBERO version, not a `torch.Tensor`; use `hasattr` guard instead of unconditional `.numpy()` |
| 2026-04-18 | Multi-server launcher | Added `--num-servers N` flag that spawns N server processes on consecutive ports and multiplexes their output with a color-coded `[S0 :5555]` prefix so all servers are visible in one terminal. Ctrl-C cleanly terminates all child processes |
| 2026-04-18 | Kill stale processes before binding | `run_multi_server` now calls `lsof` to find and kill any processes bound to the target ports before spawning child servers, preventing "Address already in use" on restart |
| 2026-04-19 | Support dynamic suite switching so one server handles all unseen tasks | Extracted `_load_task_suite` helper. Changed `run_session` to accept a `suite_cache` dict instead of fixed `tasks`/`task_suite` params. Reset command reads optional `suite_name`; if different from current, loads the new suite (with caching), closes the active env, and resets task state. `run_server` initializes the cache with the startup suite |
| 2026-04-19 | Print all suite/task options at startup and include suite in task-load log | Added `_print_task_listing()` that logs every suite from `LIBERO_CONFIGS` with all task indices from `libero_task_map`. Called once in `__main__` only when `LIBERO_WORKER` env var is not set, so multi-server children (which inherit `LIBERO_WORKER=1`) stay silent. Task-load log line now shows `[suite:idx]` so it's clear which suite is active |
| 2026-04-19 | Update env name to the one set by connecting client | When a client reset switches `suite_name`, `run_session` now also updates `cfg` and `image_key` to match the new suite's `LIBERO_CONFIGS` entry. `run_session` return type extended to include final suite name; `run_server` uses it as `env_key` in `save_video` so videos are tagged with the active suite |
| 2026-04-20 | Return suite_name in get_tasks so client knows which suite is active without injecting it manually | Added `suite_name` field to each task dict in the `get_tasks` response using `current_suite_name` |

---

## `scripts/evaluate.py`

**Generated:** 2026-04-18 | `claude-sonnet-4-6`  
**Prompt:** Batch evaluation script that runs multiple model checkpoints on all seen LIBERO-10 tasks and optionally unseen tasks, collecting per-task success rates and writing CSV reports. Validate mode uses init states 0–19, test mode uses 20–39; unseen evaluation is gated by `--run-eval-on-unseen` flag.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-18 | Add batched environment inference | Replaced `run_episode` with `run_batched_episodes` that connects to N servers, resets them in parallel via `ThreadPoolExecutor`, stacks observations into a single `(N, ...)` batch for one model forward pass, and dispatches actions back to all envs simultaneously. Added `--num-envs` arg; servers are expected on consecutive ports from `--zmq-address` |
| 2026-04-18 | Fix ZMQ thread-safety crash | ZMQ sockets must live and be used in the same thread. Replaced bare `RemoteEnv`+`ThreadPoolExecutor` with `_EnvWorker`: each env gets a dedicated thread that owns its socket. Work is submitted via a queue; callers get a `Future` back. Removed the `ThreadPoolExecutor` entirely — `_EnvWorker` threads provide the parallelism |
| 2026-04-18 | Work-queue episode scheduling | Replaced fixed-batch loop with a queue so envs that finish early immediately reset to the next init state instead of waiting for batch-mates. Batch size stays at N for the whole task rather than shrinking at the end |
| 2026-04-18 | Fix progress bar granularity after batching | Switched from tqdm over batches to `tqdm(total=n_episodes)` with `pbar.update(len(batch))` so the bar still ticks once per episode regardless of batch size |
| 2026-04-18 | Early exit on LIBERO success | Break out of the episode loop immediately when `reward == 1.0` since LIBERO uses sparse binary rewards and continuing after success wastes time |
| 2026-04-18 | Add --max-episodes flag for quick smoke testing | Added optional `--max-episodes` arg that caps `init_state_idxs` to the first N states of the mode's range, so a 2-episode run can verify the pipeline without waiting for all 20 episodes per task |
| 2026-04-18 | Fail fast on unreachable env server | `_EnvWorker.__init__` now blocks until the worker thread finishes `RemoteEnv` setup and re-raises any connection error in the main thread. Previously a failed ping killed the worker thread silently and the main loop hung on reset futures that would never complete |
| 2026-04-18 | Bump reset timeout for task switches | Use the new `RemoteEnv` ping/op timeout split (ping 60s, ops 600s) and raise the reset `Future` timeout from 120s to 600s. Task transitions force the server to close+rebuild a LIBERO MuJoCo env which can exceed the old budgets when several servers rebuild at once |
| 2026-04-18 | Graceful env-death handling | `run_episodes_queue` now tolerates unresponsive servers: each `Future.result` call is wrapped so a timeout or exception marks that env dead (via a persistent `_EnvWorker.dead` flag), counts its in-flight episode as a failure, and lets the remaining live envs continue. Dead envs stay out of the pool for all subsequent tasks. Timeouts were tightened to step=30s / reset=300s so hangs are detected faster now that the run no longer dies on them |
| 2026-04-18 | Retry env-killed episodes, drop from denominator when unretriable | `run_episodes_queue` now returns `list[Optional[bool]]` where None means the episode never completed because every live env died before retry could succeed. When an env dies mid-episode its `init_state` is pushed back onto the work queue so a surviving env retries it. `evaluate_on_tasks` computes the task success rate over completed episodes only (None entries excluded from both numerator and denominator) and logs how many were skipped due to env failures |
| 2026-04-18 | Stop log lines corrupting progress bars | Added `_TqdmLoggingHandler` that routes records through `tqdm.write`, installed via `basicConfig(force=True)` so warnings emitted mid-task (e.g. env death) no longer break the active tqdm bar |
| 2026-04-18 | Save successes/attempts per task | `evaluate_on_tasks` now returns `{task: {"successes", "attempts"}}` instead of just the rate. CSV columns changed to `<task>_successes` / `<task>_attempts` pairs plus `total_successes`, `total_attempts`, and `avg_success_rate` (mean of per-task rates, equal-weighted). New `build_csv_row` helper flattens the counts for the writer |
| 2026-04-18 | Parallel worker startup with shared deadline | `_EnvWorker.__init__` is now non-blocking; `wait_ready(timeout_s)` does the blocking check. `_make_env_workers` starts all threads simultaneously then calls `wait_ready` with a shared deadline so total startup time = min(all_ready, startup_timeout_s=30s). Workers that miss the deadline are marked dead and skipped; evaluation starts immediately with whoever connected |
| 2026-04-19 | Resume from partial results, progressive CSV writes, --restart flag | Added `_load_progress` to detect completed/partial checkpoint rows from an existing CSV (using empty-string sentinels). Added `_init_csv` to create the file with headers before any checkpoint finishes. Extended `build_csv_row` with `all_task_names` param so partial rows can emit empty strings for unevaluated tasks. `write_csv` now tolerates missing keys and accepts `verbose=False` for silent mid-run writes. `evaluate_on_tasks` gains `partial_results` (skip already-done tasks) and `on_task_done` callback (write partial CSV row after each task). `main` restructured: CSV created at startup, progress loaded/skipped before the loop, row appended after each checkpoint, partial row written after each task. Same logic applied to the unseen eval block. `--restart` flag to wipe progress |
| 2026-04-19 | Add unseen tasks and suite switching support | Populated `UNSEEN_TASKS` with 7 tasks from `libero_goal` (idx 3,5,9), `libero_object` (idx 4,7), and `libero_spatial` (idx 2,7). Added `suite_name` field to task dicts and threaded it through `_EnvWorker.reset()`, `run_episodes_queue()`, and `evaluate_on_tasks()` so the server can switch suites without needing separate processes |
| 2026-04-19 | dataset_path is base dir; join with train_task_suite; inject suite_name into seen tasks | `load_model` fallback now computes the HDF5 folder as `os.path.join(cfg["dataset_path"], cfg["train_task_suite"])`. After fetching `available_tasks` from the server, `suite_name` is set to `cfg["train_task_suite"]` on each task dict so seen-task resets switch to the correct suite at evaluation start |
| 2026-04-19 | Fix stray break halting seen-task loop | Removed erroneous `break` on line 1038 that caused the checkpoint loop to exit immediately without evaluating any checkpoints |
| 2026-04-19 | Add suite and init_states columns to CSV | `build_csv_row`, `write_csv`, and `_init_csv` now accept `suite` and `init_states` params (e.g. `"libero_10"`, `"20-39"`) and write them as the second and third CSV columns after model. Unseen CSV records the sorted unique suite names from `UNSEEN_TASKS`. `_load_progress` unchanged (ignores non-task cols) |

---

## `scripts/test_env.py`

**Generated:** 2026-04-14 | `claude-opus-4-6`  
**Prompt:** Test script that connects to an environment and runs random actions to verify the env setup (including ZMQ bridge for LIBERO) works.

| Date | Prompt | Description |
|------|--------|-------------|
| 2026-04-14 | Change LIBERO actions to cover entire area | Replaced random actions for LIBERO with a systematic sweep that moves the end-effector through a grid of positions across the workspace, holding orientation and gripper steady, so the test exercises the full reachable area |
| 2026-04-14 | Cover all rotation angles too | Extended the LIBERO sweep to also rotate through ±rx, ±ry, ±rz (action dims 3–5) after the position sweep, so the test covers all 6 DOF of the action space |
| 2026-04-14 | Fix sweep to cover negative directions | Changed sweep to go negative first for N steps, then positive for 2N steps per axis, so the arm traverses from negative extreme through start to positive extreme instead of just going out and back to start |

---

## `scripts/rel2abs.py`

**Note:** Adapted from [X-VLA/evaluation/libero/rel2abs.py](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/rel2abs.py).  
Refined with Claude to change all HDF5 files of a given input directory to map relative actions to absolute actions.

*(No structured ledger — original file comment retained as attribution)*
