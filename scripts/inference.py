import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision


# TODO: to be completed, rough draaft
def inference(diff_model, noise_scheduler, plot_interval=0):
    # TODO: (fix to action) Start with pure gaussian noise
    x = torch.randn(4, 3, 256, 256)

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        # Predict noise
        with torch.no_grad():
            noise_pred = diff_model(x, t)

        # Denoise xt -> xt-1
        scheduler_output = noise_scheduler.step(noise_pred, t, x)
        x = scheduler_output.prev_sample

    if plot_interval != 0:
        if i % plot_interval == 0 or i == len(noise_scheduler.timesteps) - 1:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            # TODO: Plot predicted denoised action
            pred_x0 = scheduler_output.pred_original_sample
    return x
