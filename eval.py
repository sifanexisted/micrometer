import os
from tqdm import tqdm

import ml_collections

import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from utils.datapipe import create_datasets, create_dataloaders, BatchParser
from utils.model_init import (
    create_model,
    create_optimizer,
    create_train_state,
    compute_total_params,
)
from utils.checkpoint import create_checkpoint_manager, restore_checkpoint
from utils.train_eval import create_eval_step, eval_model_over_batch


def evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    model = create_model(config)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)
    # Create train state
    state = create_train_state(config, model, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create eval step function
    eval_step = create_eval_step(config, model)

    # Job name for checkpoint manager
    job_name = f"{config.model.model_name}"
    if config.dataset.train_samples < 40000:
        job_name = job_name + f"_sample_{int(config.dataset.train_samples / 1000)}k"

    if config.training.max_hours < 72:
        job_name = job_name + f"_hr_{config.training.max_hours}"

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore checkpoint
    state = restore_checkpoint(ckpt_mngr, state)

    # Create dataloaders
    train_dataset, test_dataset = create_datasets(config)
    train_loader, test_loader = create_dataloaders(config, train_dataset, test_dataset)

    # Create batch parser
    sample_batch = next(iter(train_loader))
    batch_parser = BatchParser(config, sample_batch)

    # Evaluate the model
    eval_metrics = {"l1_error": [], "l2_error": [], "rmse": []}
    for batch in tqdm(test_loader):
        batch = jax.tree.map(jnp.array, batch)  # Move batch to gpu devices
        batch = batch_parser.query_all(batch)
        batch = multihost_utils.host_local_array_to_global_array(
            batch, mesh, P("batch")  # Distribute batch across devices
        )

        batch_metrics, pred, y = eval_model_over_batch(
            config, state, batch, mesh, eval_step
        )

        for key in eval_metrics:
            eval_metrics[key].append(batch_metrics[key])  # metrics for all batches

    # Compute mean metrics over test dataset
    metrics = {key: jnp.array(value).mean() for key, value in eval_metrics.items()}

    print(
        "l2_error: {:.3e}, l1_error: {:.3e}, rmse: {:.3e}".format(
            metrics["l2_error"], metrics["l1_error"], metrics["rmse"]
        )
    )

    return None
