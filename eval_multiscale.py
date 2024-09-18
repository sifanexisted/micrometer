import os
from functools import partial
from tqdm import tqdm

import h5py

import numpy as np
from einops import rearrange

import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from torch.utils.data import DataLoader

from utils.datapipe import (
    BatchParser,
    BaseDataset,
)
from utils.model_init import (
    create_model,
    create_optimizer,
    create_train_state,
    compute_total_params,
)
from utils.checkpoint import create_checkpoint_manager, restore_checkpoint
from utils.train_eval import create_eval_fn


def evaluate(config):
    # Initialize model
    model = create_model(config)

    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)
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

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), config.model.model_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore checkpoint
    state = restore_checkpoint(ckpt_mngr, state)

    # Create eval step function
    eval_fn = create_eval_fn(config, model)
    eval_fn = jax.jit(
        partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(), P("batch")),
            out_specs=(P("batch"), P("batch")),
        )(eval_fn)
    )

    # Load test dataset for homogenization
    data_path = config.dataset.data_path
    test_suffixes = config.dataset.test_files
    for suffix in test_suffixes:
        print("Processing {}...".format(suffix))

        test_input_key = ["cmme_ms_inputs"]
        test_label_key = [f"cmme_ms_labels_{suffix}"]
        test_output_key = test_input_key  # Dummy output key

        test_input_file = [data_path + test_input_key[0] + ".mat"]
        test_label_file = [data_path + test_label_key[0] + ".mat"]
        test_output_file = test_input_file  # Dummy output file

        test_dataset = BaseDataset(
            test_input_file,
            test_output_file,
            test_label_file,
            test_input_key,
            test_output_key,
            test_label_key,
            downsample_factor=1,
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
        print('Evaluating... {}'.format(suffix))
        pred_list = []
        for batch in tqdm(test_loader):
            batch = jax.tree.map(jnp.array, batch)
            batch = batch_parser.query_all(batch)
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )

            sub_pred_list = []
            chunk_size = batch[0].shape[1] // config.eval.num_eval_chunks
            for chunk in range(config.eval.num_eval_chunks):
                # None that the shape of coords is (num_devices, h * w, 2)
                sub_batch = (
                    batch[0][:, chunk * chunk_size: (chunk + 1) * chunk_size],
                    batch[1],
                    batch[2][:, chunk * chunk_size: (chunk + 1) * chunk_size],
                )

                sub_batch = multihost_utils.host_local_array_to_global_array(
                    sub_batch, mesh, P("batch")
                )
                sub_pred, sub_y = eval_fn(state.params, sub_batch)
                sub_pred_list.append(sub_pred)

            pred = jnp.concatenate(sub_pred_list, axis=1)

            pred = rearrange(pred, "b (h w) c -> b c h w", h=256, w=256)
            pred_list.append(np.array(pred, dtype=np.float32))

        # Save predictions as h5 file
        preds = np.vstack(pred_list)

        # ms_dir = os.path.join(os.getcwd(), "ms_preds")
        ms_dir = data_path
        os.makedirs(ms_dir, exist_ok=True)  # Create the directory if it doesn't exist
        ms_path = os.path.join(ms_dir, f"cmme_ms_preds_{suffix}.h5")

        with h5py.File(ms_path, "w") as h5f:
            h5f.create_dataset(f"cmme_ms_preds_{suffix}", data=preds)
