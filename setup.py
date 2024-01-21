from setuptools import setup, find_packages

setup(
    name="cnn-vae-project",
    version="0.1.0",
    description="Convolutional Neural Network Variational Autoencoder",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/Riffe007/cnn-vae-2",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "keras",
        "numpy",
        "matplotlib",
    ],
    python_requires='>=3.6',
    # Additional classifiers can be added here
)
