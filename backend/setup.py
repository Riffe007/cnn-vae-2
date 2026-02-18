from setuptools import find_packages, setup

setup(
    name="cnn-vae-project",
    version="0.4.0",
    description="Convolutional Variational Autoencoder built with TensorFlow/Keras",
    author="Riffe007",
    url="https://github.com/Riffe007/cnn-vae-2",
    packages=find_packages(include=["backend", "backend.*"]),
    install_requires=[
        "tensorflow>=2.15,<3.0",
        "numpy>=1.24,<3.0",
        "matplotlib>=3.8,<4.0",
        "datasets>=2.18,<5.0",
        "fastapi>=0.115,<1.0",
        "uvicorn>=0.30,<1.0",
        "pillow>=10.4,<12.0",
    ],
    python_requires=">=3.10",
)
