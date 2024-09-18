import jax
import jax.numpy as jnp

import flax.linen as nn

from einops import rearrange, repeat

from .vit import Encoder, CrossAttnBlock
from .fno2d import FourierStage

from typing import Optional, Callable, Dict


class FNOEncoder(nn.Module):
    emb_dim: int = 64
    modes1: int = 32
    modes2: int = 32
    depth: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.emb_dim)(x)

        for _ in range(self.depth):
            x = FourierStage(
                emb_dim=self.emb_dim,
                modes1=self.modes1,
                modes2=self.modes2,
            )(x)

        return x


def compute_dist_weights(coords, grid, eps):
    """
    coords: (b, n, 2)
    grid: (h * w, 2)
    eps: float
    output: (n, h * w)
    """
    # Compute distance between coords and grid
    d2 = ((coords[:, jnp.newaxis, :] - grid[jnp.newaxis, :, :]) ** 2).sum(axis=2)
    # Compute weights using exponential kernel
    w = jnp.exp(-eps * d2) / jnp.exp(-eps * d2).sum(axis=1, keepdims=True)
    return w


class CViT(nn.Module):
    patch_size: tuple = (16, 16)
    grid_size: tuple = (256, 256)
    fourier_emb_dim: int = 128
    fourier_modes: int = 32
    fourier_depth: int = 3
    emb_dim: int = 256
    num_heads: int = 16
    depth: int = 12
    dec_emb_dim: int = 256
    dec_num_heads: int = 8
    dec_depth: int = 1
    mlp_ratio: int = 1
    out_dim: int = 1
    eps: float = 1e5
    layer_norm_eps: float = 1e-5
    model_name: Optional[str] = None

    def setup(self):
        # Create grid and latents
        n_x, n_y = self.grid_size[0], self.grid_size[1]
        x = jnp.linspace(0, 1, n_x)
        y = jnp.linspace(0, 1, n_y)
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        self.grid = jnp.hstack([xx.flatten()[:, None], yy.flatten()[:, None]])
        # self.latents = self.param("latents", normal(), (n_x, n_y, self.grid_emb_dim))

    @nn.compact
    def __call__(self, x, coords):
        b, h, w, c = x.shape

        # Lift inputs to latent space by FNO layers
        x = FNOEncoder(
            depth=self.fourier_depth,
            emb_dim=self.fourier_emb_dim,
            modes1=self.fourier_modes,
            modes2=self.fourier_modes,
        )(x)

        # If input size is different from grid size, resize input to grid size
        if (h, w) != self.grid_size:
            x = jax.image.resize(x, (b, *self.grid_size, c), method="bilinear")

        w = compute_dist_weights(coords, self.grid, self.eps)  # (n, h * w)
        w = rearrange(w, "n (h w) -> n h w", h=self.grid_size[0], w=self.grid_size[1])

        # Interpolate latents using weights
        coords = jnp.einsum("bhwc,nhw->bnc", x, w)
        coords = nn.Dense(self.dec_emb_dim)(coords)
        coords = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)

        x = Encoder(
            self.patch_size,
            self.emb_dim,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.layer_norm_eps,
        )(x)
        x = nn.Dense(self.dec_emb_dim)(x)

        # Decoder
        for _ in range(self.dec_depth):
            x = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        x = nn.Dense(self.out_dim)(x)

        return x
