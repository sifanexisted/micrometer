import ml_collections

from configs import models


def get_config(model):
    """Get the hyperparameter configuration for a specific model."""
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shape for initializing Flax models
    config.x_dim = [2, 256, 256, 7]
    config.coords_dim = [1024, 2]  # Only for initializing CViT model

    # Training or evaluation
    config.mode = "eval_homo"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "micrometer"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/lus/eagle/projects/Micrometer/CMME/homo/"

    dataset.test_files = ["p1", "p2", "p3", "p4", "p5"]
    dataset.downsample_factor = 2
    dataset.train_batch_size = 16  # Per device
    dataset.test_batch_size = 2  # Per device
    dataset.train_samples = 40000  # Use all data for training
    dataset.test_samples = 2000  # Use all data for testing
    dataset.num_workers = 8

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-3
    lr.decay_rate = 0.9
    lr.transition_steps = 5000
    lr.warmup_steps = 2000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 2 * 10 ** 5
    training.max_hours = 72
    training.num_query_points = 4096  # Only used for CViTDataset
    training.resume = False

    # Evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.num_eval_chunks = 2

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.num_keep_ckpts = 2

    return config
