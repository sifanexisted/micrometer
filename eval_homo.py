import os
from tqdm import tqdm

import ml_collections

import h5py

import numpy as np
from einops import rearrange

import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from torch.utils.data import DataLoader

from utils.datapipe import (
    BatchParser,
    PlainDataset,
    BaseDataset,
)
from utils.model_init import (
    create_model,
    create_optimizer,
    create_train_state,
    compute_total_params,
)
from utils.checkpoint import create_checkpoint_manager, restore_checkpoint
from utils.train_eval import create_eval_step, eval_model_over_batch
from utils.postprocess import compute_stress


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

    # Load test dataset for homogenization
    data_path = config.dataset.data_path
    test_suffixes = config.dataset.test_files
    total_error_list = []
    for suffix in test_suffixes:
        print("Processing {}...".format(suffix))

        test_input_key = ["cmme_homo_inputs"]
        test_output_key = [f"cmme_homo_outputs_{suffix}"]
        test_label_key = [f"cmme_homo_labels_{suffix}"]

        test_input_file = [data_path + test_input_key[0] + ".mat"]
        test_output_file = [data_path + test_output_key[0] + ".mat"]
        test_label_file = [data_path + test_label_key[0] + ".mat"]

        test_dataset = BaseDataset(
            test_input_file,
            test_output_file,
            test_label_file,
            test_input_key,
            test_output_key,
            test_label_key,
            downsample_factor=config.dataset.downsample_factor,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.dataset.test_batch_size * num_devices,
            num_workers=config.dataset.num_workers,
            shuffle=False,
            drop_last=False,
        )

        # Create batch parser
        sample_batch = next(iter(test_loader))
        batch_parser = BatchParser(config, sample_batch)

        # Evaluate
        pred_list = []

        for batch in tqdm(test_loader):
            batch = jax.tree.map(jnp.array, batch)
            batch = batch_parser.query_all(batch)
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )

            batch_metrics, pred, y = eval_model_over_batch(
                config, state, batch, mesh, eval_step
            )

            pred = rearrange(pred, "b (h w) c -> b c h w", h=256, w=256)
            pred_list.append(np.array(pred))

        # Save predictions as h5 file
        preds = np.vstack(pred_list)

        homo_dir = os.path.join(os.getcwd(), "homo_preds")
        os.makedirs(homo_dir, exist_ok=True)  # Create the directory if it doesn't exist
        homo_path = os.path.join(homo_dir, f"cmme_homo_preds_{suffix}.h5")

        # Write the predictions to the HDF5 file
        with h5py.File(homo_path, "w") as h5f:
            h5f.create_dataset(f"cmme_homo_preds_{suffix}", data=preds)

        # Load
        pred_key = [f"cmme_homo_preds_{suffix}"]
        pred_file = [homo_path]

        plain_dataset = PlainDataset(
            test_input_file,
            pred_file,
            test_label_file,
            test_input_key,
            pred_key,
            test_label_key,
            downsample_factor=config.dataset.downsample_factor,
        )

        plain_loader = DataLoader(
            plain_dataset,
            batch_size=250,
            num_workers=config.dataset.num_workers,
            shuffle=False,
            drop_last=False,
        )

        # Compute average stress
        print(f"Compute average stress for material: {suffix}")
        v_avg_list = []
        E_avg_list = []
        for batch in tqdm(plain_loader):
            batch = jax.tree.map(np.array, batch)
            inputs, outputs, labels = batch

            # Downsample
            inputs = inputs[:, :, ::2, ::2]

            v_avg, E_avg = compute_stress(inputs, outputs, labels)
            v_avg_list.append(v_avg.flatten())
            E_avg_list.append(E_avg.flatten())

        # Save average stress as numpy file
        pred_avg = np.stack([E_avg_list, v_avg_list])
        np.save(homo_dir + f"/cmme_homo_pred_avg_{suffix}.npy", pred_avg)

        # Compute relative error
        with h5py.File(data_path + f"cmme_homo_avg_{suffix}.mat", "r") as f:
            ref_avg = np.array(f[f"cmme_homo_avg_{suffix}"])
        ref_avg = ref_avg.reshape(2, 5, -1)

        ref_avg_mean = ref_avg.mean(axis=2)
        pred_avg_mean = pred_avg.mean(axis=2)

        rel_error = np.abs(ref_avg_mean - pred_avg_mean) / np.abs(ref_avg_mean)
        print(f"Relative E error: ", rel_error[0])
        print(f"Relative nu error: ", rel_error[1])

        total_error = rel_error.mean()
        total_error_list.append(total_error)
        print(f"Total error: ", total_error)

    total_error_list = np.array(total_error_list)
    print(f"error mean: {total_error_list.mean()}")
    print(f"error std: {total_error_list.std()}")

