import os
import json

import time
import ml_collections
import wandb

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
from utils.checkpoint import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)
from utils.train_eval import create_train_step, create_eval_step, eval_model_over_batch


def train_and_evaluate(config: ml_collections.ConfigDict):
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

    # Create train and eval step functions
    train_step = create_train_step(config, model)
    eval_step = create_eval_step(config, model)

    # Create dataloaders
    train_dataset, test_dataset = create_datasets(config)
    train_loader, test_loader = create_dataloaders(config, train_dataset, test_dataset)

    # Create batch parser
    sample_batch = next(iter(train_loader))
    batch_parser = BatchParser(config, sample_batch)

    # Define job name for checkpoint manager and W&B
    job_name = f"{config.model.model_name}"
    if config.dataset.train_samples < 40000:
        job_name = job_name + f"_sample_{int(config.dataset.train_samples / 1000)}k"

    if config.training.max_hours < 72:
        job_name = job_name + f"_hr_{config.training.max_hours}"

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(os.getcwd(), job_name, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        # Initialize W&B
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=job_name, config=config)

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore previous checkpoint if resume training
    if config.training.resume:
        state = restore_checkpoint(ckpt_mngr, state)

    # Training loop
    start_training_time = time.time()  # Start the time
    last_loss = 1.0
    rng_key = jax.random.PRNGKey(0)
    for epoch in range(10000):
        start_time = time.time()
        for batch in train_loader:
            rng_key, subkey = jax.random.split(rng_key)
            batch = jax.tree.map(jnp.array, batch)
            batch = batch_parser.random_query(batch, rng_key=subkey)
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )
            state, loss = train_step(state, batch)

        # Logging
        if epoch % config.logging.log_interval == 0:
            # Evaluate model
            eval_metrics = {"l1_error": [], "l2_error": [], "rmse": []}
            for batch in iter(test_loader):
                batch = jax.tree.map(jnp.array, batch)
                batch = batch_parser.query_all(batch)
                batch = multihost_utils.host_local_array_to_global_array(
                    batch, mesh, P("batch")
                )

                batch_metrics, _, _ = eval_model_over_batch(
                    config, state, batch, mesh, eval_step
                )

                for key in eval_metrics:
                    eval_metrics[key].append(
                        batch_metrics[key]
                    )  # metrics for all batches

            # Compute mean metrics over test dataset
            metrics = {
                key: jnp.array(value).mean() for key, value in eval_metrics.items()
            }

            # Log metrics
            step = int(state.step)
            loss = loss.item()
            end_time = time.time()
            log_dict = {"loss": loss, "lr": lr(step), **metrics}

            if jax.process_index() == 0:
                wandb.log(log_dict, step)  # Log metrics to W&B
                print(
                    "step: {}, loss: {:.3e}, l1_error: {:.3e}, l2_error: {:.3e}, rmse: {:.3e}, time: {:.3e}".format(
                        step,
                        loss,
                        metrics["l1_error"],
                        metrics["l2_error"],
                        metrics["rmse"],
                        end_time - start_time,
                    )
                )

            # If loss blowup, restart training from the last checkpoint
            if loss >= last_loss * 3:
                print("Loss blowup detected, reverting to last checkpoint")
                state = restore_checkpoint(ckpt_mngr, state)
                continue

            # Save checkpoints
            if loss < 1.1 * last_loss:
                save_checkpoint(ckpt_mngr, state)
                # Update the best loss
                last_loss = loss

            # Break if training has reached the maximum number of steps or max hours
            elapsed_time = (time.time() - start_training_time) / 3600
            if (
                step >= config.training.max_steps
                or elapsed_time > config.training.max_hours
            ):
                break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()
