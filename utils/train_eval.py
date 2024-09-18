from functools import partial

from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax import lax, jit, vmap, pmap, random
from jax.flatten_util import ravel_pytree
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


def compute_lp_norms(pred, y, ord=2):
    """
    pred: (b, h * w, c)
    y: (b, h * w, c)
    """

    diff_norms = jnp.linalg.norm(pred - y, axis=1, ord=ord, keepdims=True)
    y_norms = jnp.linalg.norm(y, axis=1, ord=ord, keepdims=True)
    lp_error = (diff_norms / y_norms).mean()

    return diff_norms, y_norms, lp_error


class PatchHandler:
    def __init__(self, inputs, patch_size):
        self.patch_size = patch_size

        _, self.height, self.width, self.channel = inputs.shape

        self.patch_height, self.patch_width = (
            self.height // self.patch_size[0],
            self.width // self.patch_size[1],
        )

    def merge_patches(self, x):
        batch, _, _ = x.shape
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height,
                self.patch_width,
                self.patch_size[0],
                self.patch_size[1],
                -1,
            ),
        )
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(
            x,
            (
                batch,
                self.patch_height * self.patch_size[0],
                self.patch_width * self.patch_size[1],
                -1,
            ),
        )
        return x


def create_eval_fn(config, model):
    # Single device evaluation function
    # For ViT models, we need to create a patch handler to merge patches
    if config.model.model_name.lower().startswith("vit"):
        patch_handler = PatchHandler(jnp.ones(config.x_dim), config.model.patch_size)

    def eval_fn(params, batch):
        if config.model.model_name.lower().startswith("cvit"):
            coords, x, y = batch
            coords = jnp.squeeze(coords)

            pred = vmap(model.apply, (None, None, 0), out_axes=2)(
                params, x, coords[:, None, :]
            )
            pred = jnp.squeeze(
                pred
            )  # we need to squeeze out the extra dimension due to vmap

        else:  # For all other models
            x, y = batch
            pred = model.apply(params, x)

            # For ViT models, we need to merge the patches
            if config.model.model_name.lower().startswith("vit"):
                pred = patch_handler.merge_patches(pred)  # （B, H, W, C）

        return pred, y

    return eval_fn


def create_train_step(config, model):
    eval_fn = create_eval_fn(config, model)
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P()),
    )
    def train_step(state, batch):
        # define loss function, for CViT model we need to pass query coordinates and use vmap
        def loss_fn(params):
            pred, y = eval_fn(params, batch)
            loss = jnp.mean((y - pred) ** 2)
            return loss

        # Compute gradients and update parameters
        grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(state.params)
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def create_eval_step(config, model):
    eval_fn = create_eval_fn(config, model)

    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P("batch"), P("batch")),
    )
    def eval_step(state, batch):
        pred, y = eval_fn(state.params, batch)
        pred = pred.reshape(y.shape)
        if not config.model.model_name.lower().startswith("cvit"):
            pred = rearrange(pred, "b h w c -> b (h w) c")
            y = rearrange(y, "b h w c -> b (h w) c")

        l1_num, l1_denom, _ = compute_lp_norms(pred, y, ord=1)
        l2_num, l2_denom, _ = compute_lp_norms(pred, y, ord=2)
        rmse = jnp.sqrt(jnp.mean((pred - y) ** 2, axis=1))

        l1_num = lax.pmean(l1_num, "batch")
        l1_denom = lax.pmean(l1_denom, "batch")
        l2_num = lax.pmean(l2_num, "batch")
        l2_denom = lax.pmean(l2_denom, "batch")
        rmse = lax.pmean(rmse, "batch")

        metrics = {
            "l1_num": l1_num,
            "l1_denom": l1_denom,
            "l2_num": l2_num,
            "l2_denom": l2_denom,
            "rmse": rmse,
        }

        return metrics, pred, y

    return eval_step


@jax.jit
def merge_metrics(chunk_metrics):
    for key in chunk_metrics:
        chunk_metrics[key] = jnp.concatenate(chunk_metrics[key], axis=1)

    l1_num = jnp.linalg.norm(jnp.array(chunk_metrics["l1_num"]), ord=1, axis=1)
    l1_denom = jnp.linalg.norm(jnp.array(chunk_metrics["l1_denom"]), ord=1, axis=1)
    l2_num = jnp.linalg.norm(jnp.array(chunk_metrics["l2_num"]), ord=2, axis=1)
    l2_denom = jnp.linalg.norm(jnp.array(chunk_metrics["l2_denom"]), ord=2, axis=1)

    with jax.spmd_mode("allow_all"):
        l1_error = jnp.mean(l1_num / l1_denom)
        l2_error = jnp.mean(l2_num / l2_denom)
        rmse = jnp.sqrt(jnp.mean(jnp.array(chunk_metrics["rmse"]) ** 2))

    return {
        "l1_error": l1_error,
        "l2_error": l2_error,
        "rmse": rmse,
    }


def eval_model_over_batch(config, state, batch, mesh, eval_step):
    if not config.model.model_name.lower().startswith("cvit"):
        # Evaluate on the entire batch
        batch_metrics, pred, y = eval_step(state, batch)

        with jax.spmd_mode("allow_all"):
            l1_error = jnp.mean(batch_metrics["l1_num"] / batch_metrics["l1_denom"])
            l2_error = jnp.mean(batch_metrics["l2_num"] / batch_metrics["l2_denom"])
            rmse = jnp.mean(batch_metrics["rmse"])

        batch_metrics = {
            "l1_error": l1_error,
            "l2_error": l2_error,
            "rmse": rmse,
        }

    else:
        # Evaluate the model over chunks of the query coordinates, to reduce GPU memory usage
        chunk_metrics = {
            "l1_num": [],
            "l1_denom": [],
            "l2_num": [],
            "l2_denom": [],
            "rmse": [],
        }
        pred_list = []
        y_list = []

        chunk_size = batch[0].shape[1] // config.eval.num_eval_chunks
        for chunk in range(config.eval.num_eval_chunks):
            # None that the shape of coords is (num_devices, h * w, 2)
            sub_batch = (
                batch[0][:, chunk * chunk_size : (chunk + 1) * chunk_size],
                batch[1],
                batch[2][:, chunk * chunk_size : (chunk + 1) * chunk_size],
            )

            sub_batch = multihost_utils.host_local_array_to_global_array(
                sub_batch, mesh, P("batch")
            )
            sub_metrics, sub_pred, sub_y = eval_step(state, sub_batch)

            for key in chunk_metrics:
                chunk_metrics[key].append(sub_metrics[key])
            pred_list.append(sub_pred)
            y_list.append(sub_y)

        # Merge metrics from chunks
        batch_metrics = merge_metrics(chunk_metrics)
        pred = jnp.concatenate(pred_list, axis=1)
        y = jnp.concatenate(y_list, axis=1)

    return batch_metrics, pred, y
