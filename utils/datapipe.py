import os
from functools import partial

import h5py
import numpy as np
from einops import rearrange, repeat

import jax
import jax.numpy as jnp

from jax import random, jit

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class PlainDataset(Dataset):
    # This dataset class is used for homogenization
    def __init__(
        self,
        input_files,
        output_files,
        label_files,
        input_keys,
        output_keys,
        label_keys,
        downsample_factor=2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.num_files = len(input_files)
        self.inputs = []
        self.outputs = []
        self.labels = []

        for input_file, input_key in zip(input_files, input_keys):
            self.inputs.append(h5py.File(input_file, "r")[input_key])

        for output_file, output_key in zip(output_files, output_keys):
            self.outputs.append(h5py.File(output_file, "r")[output_key])

        for label_file, label_key in zip(label_files, label_keys):
            self.labels.append(h5py.File(label_file, "r")[label_key][:].T)

    def __len__(self):
        # Assuming all datasets have the same length, use the length of the first one
        return len(self.inputs[0]) * self.num_files

    def __getitem__(self, index):
        # Choose an input file randomly each time
        file_idx = index // len(self.inputs[0])
        index = index % len(self.inputs[0])

        input_data = self.inputs[file_idx]
        output_data = self.outputs[file_idx]
        label_data = self.labels[file_idx]

        batch_inputs = np.array(input_data[index])
        batch_outputs = np.array(output_data[index])
        batch_labels = np.array(label_data[index])

        return batch_inputs, batch_outputs, batch_labels


class BaseDataset(PlainDataset):
    # This dataset class is used for training VIT, FNO, UNet models.
    def __init__(
        self,
        input_files,
        output_files,
        label_files,
        input_keys,
        output_keys,
        label_keys,
        downsample_factor=2,
        channel_last=True,
        log_label=True,
    ):
        super().__init__(
            input_files,
            output_files,
            label_files,
            input_keys,
            output_keys,
            label_keys,
            downsample_factor,
        )
        self.channel_last = channel_last
        self.log_label = log_label

        b, c, h, w = self.inputs[0].shape
        self.h = h // self.downsample_factor
        self.w = w // self.downsample_factor

        x_star = np.linspace(0, 1, self.h)
        y_star = np.linspace(0, 1, self.w)
        x_star, y_star = np.meshgrid(x_star, y_star, indexing="ij")

        self.grid = np.stack([x_star, y_star]).T
        self.coords = np.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])

    def __getitem__(self, index):
        # Get the original batch inputs, outputs, and labels
        batch_inputs, batch_outputs, batch_labels = super().__getitem__(index)

        if self.channel_last:
            batch_inputs = np.transpose(batch_inputs, (1, 2, 0))  # Convert to (H, W, C)
            batch_outputs = np.transpose(
                batch_outputs, (1, 2, 0)
            )  # Convert to (H, W, C)

        if self.log_label:
            # batch_labels[[0, 2]] = np.log(batch_labels[[0, 2]])
            batch_labels = np.log(batch_labels)

        # Concatenate the grid and labels to the inputs
        batch_inputs = batch_inputs[
            :: self.downsample_factor, :: self.downsample_factor
        ]
        batch_labels = repeat(batch_labels, "c -> h w c", h=self.h, w=self.w)
        batch_inputs = np.concatenate([batch_inputs, batch_labels, self.grid], axis=-1)

        batch_outputs = batch_outputs[
            :: self.downsample_factor, :: self.downsample_factor
        ]

        return batch_inputs, batch_outputs


class CViTDataset(BaseDataset):
    def __init__(
        self,
        input_files,
        output_files,
        label_files,
        input_keys,
        output_keys,
        label_keys,
        downsample_factor=2,
        num_query_points=None,
        channel_last=True,
        log_label=True,
    ):
        super().__init__(
            input_files,
            output_files,
            label_files,
            input_keys,
            output_keys,
            label_keys,
            downsample_factor,
            channel_last,
            log_label,
        )
        self.num_query_points = num_query_points

    def __getitem__(self, index):
        batch_inputs, batch_outputs = super().__getitem__(index)
        batch_outputs = rearrange(batch_outputs, "h w c -> (h w) c")

        if self.num_query_points is not None:
            query_index = np.random.choice(
                batch_outputs.shape[0], self.num_query_points, replace=False
            )
            batch_coords = self.coords[query_index]
            batch_outputs = batch_outputs[query_index]
        else:
            batch_coords = self.coords

        return batch_coords, batch_inputs, batch_outputs


def generate_paths_and_keys(data_path, split, suffixes):
    inputs_path = os.path.join(data_path, f"{split}_inputs")
    outputs_path = os.path.join(data_path, f"{split}_outputs")
    labels_path = os.path.join(data_path, f"{split}_labels")

    input_keys = [f"cmme_{split}_inputs_{suffix}" for suffix in suffixes]
    output_keys = [f"cmme_{split}_outputs_{suffix}" for suffix in suffixes]
    label_keys = [f"cmme_{split}_labels_{suffix}" for suffix in suffixes]

    input_files = [os.path.join(inputs_path, f"{key}.mat") for key in input_keys]
    output_files = [os.path.join(outputs_path, f"{key}.mat") for key in output_keys]
    label_files = [os.path.join(labels_path, f"{key}.mat") for key in label_keys]

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

    train_data = generate_paths_and_keys(data_path, "train", train_suffixes)
    test_data = generate_paths_and_keys(data_path, "test", test_suffixes)

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

    if config.dataset.test_samples < len(test_dataset):
        test_indices = torch.randperm(len(test_dataset))[: config.dataset.test_samples]
        test_dataset = Subset(test_dataset, test_indices)

    return train_dataset, test_dataset


def create_dataloaders(config, train_dataset, test_dataset):
    num_devices = jax.device_count()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size * num_devices,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.test_batch_size * num_devices,
        num_workers=config.dataset.num_workers,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, test_loader


class BatchParser:
    def __init__(self, config, batch):
        self.config = config
        self.num_query_points = config.training.num_query_points

        b, h, w, c = batch[1].shape
        x_star = jnp.linspace(0, 1, h)
        y_star = jnp.linspace(0, 1, w)
        x_star, y_star = jnp.meshgrid(x_star, y_star, indexing="ij")

        self.coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None]])

    @partial(jit, static_argnums=(0,))
    def random_query(self, batch, rng_key=None):
        batch_inputs, batch_outputs = batch
        batch_outputs = rearrange(batch_outputs, "b h w c -> b (h w) c")

        query_index = random.choice(
            rng_key, batch_outputs.shape[1], (self.num_query_points,), replace=False
        )
        batch_coords = self.coords[query_index]
        batch_outputs = batch_outputs[:, query_index]

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs

    @partial(jit, static_argnums=(0,))
    def query_all(self, batch):
        batch_inputs, batch_outputs = batch

        if self.config.model.model_name.lower().startswith("cvit"):
            batch_outputs = rearrange(batch_outputs, "b h w c -> b (h w) c")
            batch_coords = self.coords

            # Repeat the coords  across devices
            batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

            return batch_coords, batch_inputs, batch_outputs

        return batch_inputs, batch_outputs
