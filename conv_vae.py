"""Convolutional Variational Autoencoder (CNN-VAE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

INPUT_DIM = (64, 64, 3)
DEFAULT_FILTERS = (32, 64, 64, 128)
DEFAULT_KERNEL_SIZE = 4
DEFAULT_STRIDE = 2
Z_DIM = 32
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
BETA = 1.0


@dataclass(frozen=True)
class ConvBlockConfig:
    filters: int
    kernel_size: int
    stride: int


def _make_conv_blocks(
    conv_filters: Iterable[int], kernel_size: int, stride: int
) -> tuple[ConvBlockConfig, ...]:
    return tuple(ConvBlockConfig(filters=f, kernel_size=kernel_size, stride=stride) for f in conv_filters)


def _sampling(args: tuple[tf.Tensor, tf.Tensor], z_dim: int) -> tf.Tensor:
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch, z_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ConvVAE:
    """A compact CNN-VAE for image-like tensors in NHWC format."""

    def __init__(
        self,
        input_dim: tuple[int, int, int] = INPUT_DIM,
        z_dim: int = Z_DIM,
        conv_filters: tuple[int, ...] = DEFAULT_FILTERS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        stride: int = DEFAULT_STRIDE,
        learning_rate: float = LEARNING_RATE,
        beta: float = BETA,
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.conv_blocks = _make_conv_blocks(conv_filters, kernel_size, stride)
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
        self.encoder, self.decoder, self.vae = self._build()

    def _build(self) -> tuple[Model, Model, Model]:
        vae_x = layers.Input(shape=self.input_dim, name="encoder_input")
        x = vae_x
        for i, block in enumerate(self.conv_blocks):
            x = layers.Conv2D(
                filters=block.filters,
                kernel_size=block.kernel_size,
                strides=block.stride,
                activation="relu",
                padding="same",
                name=f"enc_conv_{i}",
            )(x)

        shape_before_flatten = x.shape[1:]
        x = layers.Flatten(name="flatten")(x)
        z_mean = layers.Dense(self.z_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.z_dim, name="z_log_var")(x)
        z = layers.Lambda(lambda args: _sampling(args, self.z_dim), name="z")([z_mean, z_log_var])
        encoder = Model(vae_x, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = layers.Input(shape=(self.z_dim,), name="z_sampling")
        x = layers.Dense(int(np.prod(shape_before_flatten)), name="dec_dense")(latent_inputs)
        x = layers.Reshape(tuple(shape_before_flatten), name="dec_reshape")(x)
        for i, block in enumerate(reversed(self.conv_blocks)):
            x = layers.Conv2DTranspose(
                filters=block.filters,
                kernel_size=block.kernel_size,
                strides=block.stride,
                activation="relu",
                padding="same",
                name=f"dec_deconv_{i}",
            )(x)
        decoder_output = layers.Conv2DTranspose(
            filters=self.input_dim[-1],
            kernel_size=3,
            activation="sigmoid",
            padding="same",
            name="decoder_output",
        )(x)
        decoder = Model(latent_inputs, decoder_output, name="decoder")

        vae_outputs = decoder(encoder(vae_x)[2])
        vae = Model(vae_x, vae_outputs, name="vae")

        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(vae_x - vae_outputs), axis=[1, 2, 3]))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = reconstruction_loss + (self.beta * kl_loss)

        vae.add_loss(total_loss)
        vae.add_metric(reconstruction_loss, name="reconstruction_loss")
        vae.add_metric(kl_loss, name="kl_loss")
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        return encoder, decoder, vae

    def train(
        self,
        data: np.ndarray,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        validation_split: float = 0.2,
        patience: int = 5,
    ) -> tf.keras.callbacks.History:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
        return self.vae.fit(
            data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1,
        )

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.vae.predict(x, verbose=0)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z_mean, z_log_var, z = self.encoder.predict(x, verbose=0)
        return z_mean, z_log_var, z

    def decode_latents(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.predict(z, verbose=0)

    def generate(self, num_samples: int, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
        z = np.random.normal(size=(num_samples, self.z_dim)).astype("float32")
        return self.decode_latents(z)

    def save_weights(self, filepath: str) -> None:
        self.vae.save_weights(filepath)

    def load_weights(self, filepath: str) -> None:
        self.vae.load_weights(filepath)


if __name__ == "__main__":
    vae = ConvVAE()
    print("Model built successfully:", vae.vae.name)
