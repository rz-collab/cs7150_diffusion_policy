import torch
import torch.nn as nn

"""
Diffusion Denoiser/Noise Predictor

Adapted from Diffusion Policy Colab's Notebook.
"""


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, in_dim, out_dim, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_dim),
            nn.Mish(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_dim, T)
        Returns:
            (B, out_dim, T)
        """
        return self.block(x)


class FiLMConditionalLayer(nn.Module):
    """
    FiLM modulation https://arxiv.org/abs/1709.07871
    Use conditional input to predict per-channel scale and bias and
    apply this affine transformation to the input `x`
    """

    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim

        # Nonlinearity first because the conditioning input is the visual encoder output
        self.film_generator = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 2 * in_dim),
        )

    def forward(self, x, cond):
        """
        Args:
            x:    (B, in_dim, T)
            cond: (B, cond_dim)
        Returns:
            modulated_x (B, in_dim, T)
        """
        _, in_dim, _ = x.shape

        cond_embed = self.film_generator(cond)
        scale = cond_embed[:, 0:in_dim].unsqueeze(-1)  # (B, in_dim, 1)
        bias = cond_embed[:, in_dim:].unsqueeze(-1)

        modulated_x = scale * x + bias
        return modulated_x


class ConditionalUNet1DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, kernel_size, n_groups=8):
        super().__init__()

        self.conv1d_1 = Conv1dBlock(in_dim, out_dim, kernel_size, n_groups)
        self.film_layer = FiLMConditionalLayer(out_dim, cond_dim)
        self.conv1d_2 = Conv1dBlock(out_dim, out_dim, kernel_size, n_groups)

        self.skip_conv = nn.Identity()
        if in_dim != out_dim:
            self.skip_conv = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x, cond):
        """
        Args:
        - x: (B, input_dim, T)
        - cond: (B, cond_dim)

        Returns
        out (B, out_dim)
        """

        # residual connection
        out = self.conv1d_1(x)
        out = self.film_layer(out, cond)
        out = self.conv1d_2(out)

        # skip connection
        out = out + self.skip_conv(x)
        return out


class Downsample1d(nn.Module):
    """Strided Conv1d: halves T."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Transposed Conv1d: doubles T."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        1D UNet for diffusion noise prediction (convolution along the time axis),
        with number of downsample/upsample levels determined by `len(down_dims)`.
        At each level, input passes through two blocks of `ConditionalUnet1DBlocks` and a downsample/upsample layer.

        Args:
            input_dim:     action dim
            cond_dim:      input condition dim
            down_dims:     Channel sizes per encoder level; length sets depth.
            kernel_size:   Conv kernel size used throughout.
            n_groups:      GroupNorm groups.

        Note: Mostly copied from Diffusion Policy Repository, moved diffusion step encoding out.
        """
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.down_modules = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalUNet1DBlock(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalUNet1DBlock(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalUNet1DBlock(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalUNet1DBlock(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i == (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalUNet1DBlock(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalUNet1DBlock(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ):
        """
        Args:
            x:              (B, T, input_dim) noisy action sequence
            cond:           (B, cond_dim)  condition
        Returns:
            (B, T, input_dim) predicted noise
        """
        # (B,T,C) -> (B,C,T)
        x = x.moveaxis(-1, -2)

        out = x
        h = []
        for _, (unet_block_1, unet_block_2, downsample) in enumerate(self.down_modules):
            out = unet_block_1(out, cond)
            out = unet_block_2(out, cond)
            h.append(out)
            out = downsample(out)

        for mid_module in self.mid_modules:
            out = mid_module(out, cond)

        for _, (unet_block_1, unet_block_2, upsample) in enumerate(self.up_modules):
            out = torch.cat((out, h.pop()), dim=1)
            out = unet_block_1(out, cond)
            out = unet_block_2(out, cond)
            out = upsample(out)

        out = self.final_conv(out)

        # (B,C,T) -> (B,T,C)
        out = out.moveaxis(-1, -2)
        return out
