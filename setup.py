from setuptools import setup

setup(
    name="cnn-vae-project",
    version="0.4.0",
    description="Convolutional Variational Autoencoder built with TensorFlow/Keras",
    author="Riffe007",
    url="https://github.com/Riffe007/cnn-vae-2",
    py_modules=["conv_vae", "config", "data_hf", "train_eval", "prompt_bank", "api_service"],
    install_requires=[
        "tensorflow>=2.15,<3.0",
        "numpy>=1.24,<3.0",
        "matplotlib>=3.8,<4.0",
        "datasets>=2.18,<4.0",
        "fastapi>=0.115,<1.0",
        "uvicorn>=0.30,<1.0",
        "pillow>=10.4,<13.0",
    ],
    python_requires=">=3.10",
)
