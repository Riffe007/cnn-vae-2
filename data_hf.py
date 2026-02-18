"""Hugging Face dataset loading utilities for CNN-VAE training."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from datasets import ClassLabel, load_dataset


def _to_float_image_array(image_obj, image_size: Tuple[int, int]) -> np.ndarray:
    arr = np.array(image_obj)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    resized = tf.image.resize(arr, image_size, method="bilinear")
    return tf.clip_by_value(resized / 255.0, 0.0, 1.0).numpy().astype("float32")


def load_hf_images(
    dataset_name: str,
    split: str = "train",
    image_column: str = "image",
    image_size: Tuple[int, int] = (64, 64),
    max_samples: int | None = None,
) -> np.ndarray:
    """Load image tensors from a Hugging Face dataset into NHWC float32 format."""
    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    images = []
    for row in dataset:
        images.append(_to_float_image_array(row[image_column], image_size))

    if not images:
        raise ValueError("No images loaded from dataset. Check split/image_column/max_samples.")

    return np.stack(images, axis=0)


def load_hf_images_and_labels(
    dataset_name: str,
    split: str,
    image_column: str,
    label_column: str,
    image_size: Tuple[int, int] = (64, 64),
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images and integer labels; also return class names when available."""
    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    labels = np.array(dataset[label_column], dtype="int32")

    label_feature = dataset.features.get(label_column)
    if isinstance(label_feature, ClassLabel):
        class_names = list(label_feature.names)
    else:
        unique_labels = sorted(set(int(v) for v in labels.tolist()))
        class_names = [str(v) for v in unique_labels]

    images = []
    for row in dataset:
        images.append(_to_float_image_array(row[image_column], image_size))

    if not images:
        raise ValueError("No images loaded from dataset. Check split/image_column/max_samples.")

    return np.stack(images, axis=0), labels, class_names
