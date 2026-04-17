# ---
# Generated: 2025-01-01 | claude-opus-4-6
# Prompt: PushTImageDataset and helper functions for loading pusht zarr data
# Modifications:
#   2026-04-14 | Prompt: Add text descriptions for diffusion model conditioning | Added descriptions_path param to PushTDataset, loads JSON descriptions and returns a random one per sample in __getitem__
#   2026-04-14 | Prompt: Accept descriptions list directly | Changed from
#               descriptions_path to a descriptions list param so the caller
#               resolves the task from task_descriptions.json.
#   2026-04-14 | Prompt: Per-sample language dropout | Moved language dropout
#               from training loop into __getitem__.  With lang_dropout_prob,
#               individual samples return "" instead of a real description,
#               letting the model learn an unconditional embedding per-sample.
# ---

# @markdown ### **Dataset**
# @markdown
# @markdown Defines `PushTImageDataset` and helper functions
# @markdown
# @markdown The dataset class
# @markdown - Load data ((image, agent_pos), action) from a zarr storage
# @markdown - Normalizes each dimension of agent_pos and action to [-1,1]
# @markdown - Returns
# @markdown  - All possible segments with length `pred_horizon`
# @markdown  - Pads the beginning and the end of each episode with repetition
# @markdown  - key `image`: shape (obs_hoirzon, 3, 96, 96)
# @markdown  - key `agent_pos`: shape (obs_hoirzon, 2)
# @markdown  - key `action`: shape (pred_horizon, 2)
# @markdown  - key `description`: a random text description of the task (if descriptions_path provided)

# TODO cleanup code, i just straigth up copied (changed dict names)
import numpy as np
import zarr
from torch.utils.data import Dataset, DataLoader
from diffusion_policy.util.normalization import (
    get_data_stats,
    normalize_data,
)


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class PushTDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        descriptions: list[str] | None = None,
        lang_dropout_prob: float = 0.0,
    ):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, "r")

        # float32, [0,1], (N,96,96,3)
        train_image_data = dataset_root["data"]["img"][:]
        train_image_data = np.moveaxis(train_image_data, -1, 1)
        # (N,3,96,96)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            "agent_pos": dataset_root["data"]["state"][:, :2],
            "actions": dataset_root["data"]["action"][:],
        }
        episode_ends = dataset_root["meta"]["episode_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data["pixels"] = train_image_data

        self.descriptions: list[str] = descriptions or []
        self.lang_dropout_prob = lang_dropout_prob

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def get_stats(self):
        """Return a dictionary mapping observation key and a dict containing its min and max"""
        return self.stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["pixels"] = nsample["pixels"][: self.obs_horizon, :]
        nsample["agent_pos"] = nsample["agent_pos"][: self.obs_horizon, :]

        # sample a random task description for text conditioning;
        # each sample independently has lang_dropout_prob chance of ""
        if self.descriptions:
            desc = self.descriptions[np.random.randint(len(self.descriptions))]
            if np.random.random() < self.lang_dropout_prob:
                desc = ""
            nsample["description"] = desc

        return nsample
