import numpy as np
import torch


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def convert_stats_from_np_to_torch(stats, device):
    for key in stats:
        if isinstance(stats[key], dict):
            convert_stats_from_np_to_torch(stats[key], device)
        elif isinstance(stats[key], np.ndarray):
            stats[key] = torch.from_numpy(stats[key]).float().to(device)
    return stats
