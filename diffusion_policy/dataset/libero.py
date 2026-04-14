from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import Dataset, ConcatDataset

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils

from typing import Literal, List

import torch
import glob
import os
import re

# This dictionary maps modality (can be rgb, low_dim, scan, or depth) to
# observation key (e.g. agentview_rgb)
# It's used internally for data normalization and memory caching behaviors in SequenceDataset, based on their modality.
# See more in https://robomimic.github.io/docs/tutorials/observations.html?highlight=observation%20modality
DEFAULT_OBS_MODALITY_KEY_MAP = {
    "low_dim": ["gripper_states", "joint_states"],
    "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
}
# Map each key to its corresponding modality for preprocessing purposes.
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": DEFAULT_OBS_MODALITY_KEY_MAP})


def _split_train_test(hdf5_path):
    """Split demos 0-44 as train, 45-49 as test. Writes mask/train and mask/test
    filter keys to the HDF5, used by get_libero_dataset to select subsets."""
    train_demo_keys = [f"demo_{i}" for i in range(0, 45)]
    test_demo_keys = [f"demo_{i}" for i in range(45, 50)]

    FileUtils.create_hdf5_filter_key(hdf5_path, train_demo_keys, key_name="train")
    FileUtils.create_hdf5_filter_key(hdf5_path, test_demo_keys, key_name="test")


def extract_language(hdf5_path):
    basename = os.path.splitext(os.path.basename(hdf5_path))[0]
    basename = re.sub(r"_demo$", "", basename)
    basename = re.sub(r"^[A-Z_]+SCENE\d+_", "", basename)
    return basename.replace("_", " ")


def get_hdf5_files_from_folders(folders: List[str]) -> List[str]:
    """
    Return all .hdf5 files from the given list of folders.

    Args:
        folders: list of directory paths to search.
    Returns:
        Sorted list of absolute paths to .hdf5 files.
    """
    paths = []
    for folder in folders:
        paths.extend(glob.glob(os.path.join(folder, "**", "*.hdf5"), recursive=True))
    return paths


class LiberoSingleTaskDataset(Dataset):
    """Loads a single LIBERO HDF5 file as a dataset, and adds the language instruction as "language" key in the sample"""

    def __init__(self, hdf5_path, obs_keys, split, seq_length):
        """
        Args:
            hdf5_path: path to the LIBERO demo HDF5 file.
            obs_keys: observation keys to include (e.g., images, joint/gripper states).
            split: 'train' (demos 0-44), 'test' (demos 45-49), or None (all demos).
            seq_length: number of consecutive timesteps per sample starting from index t.
        Returns:
            torch Dataset.
        """

        # Split first 45 demos into train and last 5 demos into test
        _split_train_test(hdf5_path)

        # See more in https://robomimic.github.io/docs/modules/dataset.html
        # This is a subclass of torch Dataset
        self.dataset = SequenceDataset(
            hdf5_path=hdf5_path,
            obs_keys=obs_keys,
            dataset_keys=["actions", "rewards"],
            filter_by_attribute=split,
            frame_stack=1,
            seq_length=seq_length,
            hdf5_cache_mode="low_dim",  # cache dataset in memory to avoid repeated file i/o
            hdf5_use_swmr=False,
            hdf5_normalize_obs=False,
            load_next_obs=False,
        )

        self.language = extract_language(hdf5_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["language"] = self.language
        return sample


def get_libero_dataset(
    hdf5_files,
    obs_keys=(
        "agentview_rgb",
        "joint_states",
        "gripper_states",
    ),
    split: Literal["train"] | Literal["test"] | None = None,
    seq_length=16,
):
    datasets = [
        LiberoSingleTaskDataset(path, obs_keys, split, seq_length)
        for path in hdf5_files
    ]
    return ConcatDataset(datasets)


def preprocess_libero_batch(batch, device, obs_horizon, image_keys=("agentview_rgb",)):
    """
    Preprocess a batch from the dataloader.
    - Move to device
    - Truncates obs to obs_horizon
    - Converts images from (B, obs_horizon, H, W, C) uint8 to (B, obs_horizon, C, H, W)
    - Normalize images from [0,255] to float [0,1]
    """

    # Move all tensors to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    for key in batch["obs"]:
        batch["obs"][key] = batch["obs"][key].to(device)

    # Preprocess observations
    for key in batch["obs"]:
        batch["obs"][key] = batch["obs"][key][:, :obs_horizon]
        if key in image_keys:
            imgs = batch["obs"][key]
            imgs = imgs.moveaxis(-1, -3).float() / 255.0  # (B, obs_horizon, C, H, W)
            batch["obs"][key] = imgs
    return batch
