import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, pmap, random
from jax.flatten_util import ravel_pytree

from flax.training import train_state

import optax

from models import CViT, UNet, ViT, FNO2d


def create_model(config):
    """Returns a model as specified in `model_configs.MODEL_CONFIGS`."""
    model_name = config.model.model_name.lower()

    if model_name.startswith("cvit"):
        model = CViT(**config.model)

    elif model_name.startswith("unet"):
        model = UNet(**config.model)

    elif model_name.startswith("vit"):
        model = ViT(**config.model)

    elif model_name.startswith("fno"):
        model = FNO2d(**config.model)

    else:
        raise ValueError(f"Unknown model name: {config.model_name}")

    return model


def create_optimizer(config):
    lr = optax.warmup_exponential_decay_schedule(
        init_value=config.lr.init_value,
        peak_value=config.lr.peak_value,
        warmup_steps=config.lr.warmup_steps,
        transition_steps=config.lr.transition_steps,
        decay_rate=config.lr.decay_rate,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.optim.clip_norm),
        optax.adamw(lr, weight_decay=config.optim.weight_decay),
    )
    return lr, tx


def create_train_state(config, model, tx, params=None):
    # Initialize the model if the params are not provided, otherwise use the provided params to create the state
    if params is None:
        x = jnp.ones(config.x_dim)
        if config.model.model_name.lower().startswith("cvit"):
            coords = jnp.ones(config.coords_dim)
            params = model.init(random.PRNGKey(config.seed), x=x, coords=coords)
        else:
            params = model.init(random.PRNGKey(config.seed), x=x)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def compute_total_params(state):
    flatten_params, _ = ravel_pytree(state.params)
    total_params = len(flatten_params)

    if total_params >= 1_000_000_000:
        print(f"Total number of parameters: {total_params / 1_000_000_000:.2f} billion")
    else:
        print(f"Total number of parameters: {total_params / 1_000_000:.2f} million")
    return total_params
