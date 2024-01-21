class Config:
    # CNN-VAE Model Parameters
    input_dim = (64, 64, 3)
    conv_filters = [32, 64, 64, 128]
    conv_kernel_sizes = [4, 4, 4, 4]
    conv_strides = [2, 2, 2, 2]
    dense_size = 1024
    z_dim = 32
    epochs = 1
    batch_size = 32

    # Database Configuration (if applicable)
    db_uri = "your_database_uri"

    # Any other configurations
    # ...

# Usage:
# from config.config import Config
# Then, access Config.input_dim, Config.conv_filters, etc.
