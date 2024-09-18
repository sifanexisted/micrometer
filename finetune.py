import os
import json

import time
import ml_collections
import wandb

import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from utils.datapipe import create_dataloaders, BaseDataset, BatchParser
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


def generate_paths_and_keys(data_path, split, suffixes):
    input_keys = [f"cmme_{split}_inputs_{suffix}" for suffix in suffixes]
    output_keys = [f"cmme_{split}_outputs_{suffix}" for suffix in suffixes]
    label_keys = [f"cmme_{split}_labels_{suffix}" for suffix in suffixes]

    input_files = [os.path.join(data_path, f"{key}.mat") for key in input_keys]
    output_files = [os.path.join(data_path, f"{key}.mat") for key in output_keys]
    label_files = [os.path.join(data_path, f"{key}.mat") for key in label_keys]

    return {
        "input_files": input_files,
        "output_files": output_files,
        "label_files": label_files,
        "input_keys": input_keys,
        "output_keys": output_keys,
        "label_keys": label_keys,
    }


def create_datasets(config):
    data_path = config.dataset.data_path

    train_suffixes = config.dataset.train_files
    test_suffixes = config.dataset.test_files

    train_data = generate_paths_and_keys(data_path, "tl_train", train_suffixes)
    test_data = generate_paths_and_keys(data_path, "tl_test", test_suffixes)

    train_dataset = BaseDataset(
        train_data["input_files"],
        train_data["output_files"],
        train_data["label_files"],
        train_data["input_keys"],
        train_data["output_keys"],
        train_data["label_keys"],
        downsample_factor=config.dataset.downsample_factor,
    )
    test_dataset = BaseDataset(
        test_data["input_files"],
        test_data["output_files"],
        test_data["label_files"],
        test_data["input_keys"],
        test_data["output_keys"],
        test_data["label_keys"],
        downsample_factor=config.dataset.downsample_factor,
    )

    if config.dataset.train_samples < len(train_dataset):
        train_indices = torch.randperm(len(train_dataset))[
            : config.dataset.train_samples
        ]
        train_dataset = Subset(train_dataset, train_indices)

    return train_dataset, test_dataset


def restore_pretrained_params(config: ml_collections.ConfigDict, model, tx):
    state = create_train_state(config, model, tx)

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.model.model_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore checkpoint
    state = restore_checkpoint(ckpt_mngr, state)
    params = state.params

    return params


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    model = create_model(config)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Restore pretrained params and create train state for finetuning
    if config.job == "from_scratch":
        params = None
    elif config.job == "from_pretrained":
        print("Restoring pretrained params...")
        params = restore_pretrained_params(config, model, tx)

    state = create_train_state(config, model, tx, params=params)
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

    # Create checkpoint manager
    work_dir = os.path.join(
        os.getcwd(),
        config.model.model_name,
        f"{config.mode}_sample_{config.dataset.train_samples}_" + config.job,
    )
    ckpt_path = os.path.join(work_dir, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Save config
    config_dict = config.to_dict()
    config_path = os.path.join(work_dir, "config.json")
    with open(config_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)

    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(
        project=wandb_config.project,
        name=f"{config.model.model_name}_{config.mode}_sample_{config.dataset.train_samples}_{config.job}",
        config=config,
    )

    # Zero shot evaluation
    metrics = {"l1_error": [], "l2_error": [], "rmse": []}
    for batch in test_loader:
        batch = jax.tree.map(jnp.array, batch)
        batch = batch_parser.query_all(batch)
        batch = multihost_utils.host_local_array_to_global_array(
            batch, mesh, P("batch")
        )
        batch_metrics, _, _ = eval_model_over_batch(
            config, state, batch, mesh, eval_step
        )

        for key in metrics.keys():
            metrics[key].append(batch_metrics[key])
    # Compute mean metrics over test dataset
    metrics = {key: jnp.array(value).mean().item() for key, value in metrics.items()}
    print(
        f"Zero shot evaluation: l1_error: {metrics['l1_error']: .3e}, l2_error: {metrics['l2_error']: .3e}, rmse: {metrics['rmse']: .3e}"
    )

    # Fine-tuning
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
            metrics = {"l1_error": [], "l2_error": [], "rmse": []}
            for batch in test_loader:
                batch = jax.tree.map(jnp.array, batch)
                batch = batch_parser.query_all(batch)
                batch = multihost_utils.host_local_array_to_global_array(
                    batch, mesh, P("batch")
                )
                batch_metrics, _, _ = eval_model_over_batch(
                    config, state, batch, mesh, eval_step
                )

                for key in metrics.keys():
                    metrics[key].append(batch_metrics[key])
            # Compute mean metrics over test dataset
            metrics = {
                key: jnp.array(value).mean().item() for key, value in metrics.items()
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
            if step >= config.training.max_steps:
                break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()
