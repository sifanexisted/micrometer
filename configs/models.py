import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


##################################
########## CViT models ###########
##################################


@_register
def get_cvit_test_config():
    # need a batch size of 8 for this model when running on GPUs of 40GB memory
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-L-16-dec_depth-4"
    config.patch_size = (16, 16)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 96
    config.fourier_modes = 64
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 12
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config

@_register
def get_cvit_h_8_config():
    # need a batch size of 8 for this model when running on GPUs of 40GB memory
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-H-8"
    config.patch_size = (16, 16)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 64
    config.fourier_modes = 64
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 8
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config

@_register
def get_cvit_h_16_config():
    # need a batch size of 8 for this model when running on GPUs of 40GB memory
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-H-16"
    config.patch_size = (16, 16)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 96
    config.fourier_modes = 64
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 12
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


@_register
def get_cvit_l_8_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-L-8"
    config.patch_size = (8, 8)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 64
    config.fourier_modes = 64
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 8
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


@_register
def get_cvit_l_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-L-16"
    config.patch_size = (16, 16)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 64
    config.fourier_modes = 64
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 8
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


@_register
def get_cvit_b_8_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-B-8"
    config.patch_size = (8, 8)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 64
    config.fourier_modes = 32
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 6
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


@_register
def get_cvit_b_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-B-16"
    config.patch_size = (16, 16)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 64
    config.fourier_modes = 32
    config.emb_dim = 512
    config.dec_emb_dim = 512
    config.depth = 6
    config.dec_depth = 2
    config.num_heads = 16
    config.dec_num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


@_register
def get_cvit_s_8_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-S-8"
    config.patch_size = (8, 8)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 32
    config.fourier_modes = 32
    config.emb_dim = 256
    config.dec_emb_dim = 256
    config.depth = 4
    config.dec_depth = 2
    config.num_heads = 8
    config.dec_num_heads = 16
    config.mlp_ratio = 1
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


@_register
def get_cvit_s_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT-S-16"
    config.patch_size = (16, 16)
    config.grid_size = (256, 256)
    config.fourier_depth = 4
    config.fourier_emb_dim = 32
    config.fourier_modes = 32
    config.emb_dim = 256
    config.dec_emb_dim = 256
    config.depth = 4
    config.dec_depth = 2
    config.num_heads = 8
    config.dec_num_heads = 16
    config.mlp_ratio = 1
    config.out_dim = 9
    config.eps = 1e5
    config.layer_norm_eps = 1e-5
    return config


##################################
########## ViT models ###########
##################################


@_register
def get_vit_l_8_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-L-8"
    config.patch_size = (8, 8)
    config.emb_dim = 768
    config.depth = 18
    config.num_heads = 12
    config.mlp_ratio = 2
    config.out_dim = 9
    config.layer_norm_eps = 1e-6
    return config


@_register
def get_vit_l_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-L-16"
    config.patch_size = (16, 16)
    config.emb_dim = 768
    config.depth = 18
    config.num_heads = 12
    config.mlp_ratio = 2
    config.out_dim = 9
    config.layer_norm_eps = 1e-6
    return config


@_register
def get_vit_b_8_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-B-8"
    config.patch_size = (8, 8)
    config.emb_dim = 512
    config.depth = 12
    config.num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.layer_norm_eps = 1e-6
    return config


@_register
def get_vit_b_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-B-16"
    config.patch_size = (16, 16)
    config.emb_dim = 512
    config.depth = 12
    config.num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 9
    config.layer_norm_eps = 1e-6
    return config


@_register
def get_vit_s_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-S-16"
    config.patch_size = (16, 16)
    config.emb_dim = 384
    config.depth = 6
    config.num_heads = 6
    config.mlp_ratio = 2
    config.out_dim = 9
    config.layer_norm_eps = 1e-6
    return config


@_register
def get_vit_s_8_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT-S-8"
    config.patch_size = (8, 8)
    config.emb_dim = 384
    config.depth = 6
    config.num_heads = 6
    config.mlp_ratio = 2
    config.out_dim = 9
    config.layer_norm_eps = 1e-6
    return config


##################################
########## FNO models ###########
##################################


@_register
def get_fno_32_16m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-32-16m"
    config.emb_dim = 32
    config.modes1 = 16
    config.modes2 = 16
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_32_32m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-32-32m"
    config.emb_dim = 32
    config.modes1 = 32
    config.modes2 = 32
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_64_16m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-64-16m"
    config.emb_dim = 64
    config.modes1 = 16
    config.modes2 = 16
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_64_32m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-64-32m"
    config.emb_dim = 64
    config.modes1 = 32
    config.modes2 = 32
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_64_64m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-64-64m"
    config.emb_dim = 64
    config.modes1 = 64
    config.modes2 = 64
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_128_16m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-128-16m"
    config.emb_dim = 128
    config.modes1 = 16
    config.modes2 = 16
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_128_32m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-128-32m"
    config.emb_dim = 128
    config.modes1 = 32
    config.modes2 = 32
    config.out_dim = 9
    config.depth = 4
    return config


@_register
def get_fno_128_64m_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO-128-64m"
    config.emb_dim = 128
    config.modes1 = 64
    config.modes2 = 64
    config.out_dim = 9
    config.depth = 4
    return config


##################################
########## UNet models ###########
##################################


@_register
def get_unet_16_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet-16"
    config.emb_dim = 16
    config.out_dim = 9
    return config


@_register
def get_unet_32_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet-32"
    config.emb_dim = 32
    config.out_dim = 9
    return config


@_register
def get_unet_64_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet-64"
    config.emb_dim = 64
    config.out_dim = 9
    return config


@_register
def get_unet_96_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet-96"
    config.emb_dim = 96
    config.out_dim = 9
    return config


@_register
def get_unet_128_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet-128"
    config.emb_dim = 128
    config.out_dim = 9
    return config
