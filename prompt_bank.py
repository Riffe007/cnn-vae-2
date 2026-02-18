"""Prompt-to-latent utilities for class-conditioned VAE generation."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")


def build_prompt_bank(
    z_mean: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, np.ndarray | list[str]]:
    class_ids = np.unique(labels)
    means = []
    stds = []
    names = []

    for class_id in class_ids:
        class_mask = labels == class_id
        class_latents = z_mean[class_mask]
        means.append(np.mean(class_latents, axis=0))
        stds.append(np.std(class_latents, axis=0) + 1e-4)
        name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(int(class_id))
        names.append(name)

    return {
        "class_ids": class_ids.astype("int32"),
        "class_names": names,
        "means": np.stack(means).astype("float32"),
        "stds": np.stack(stds).astype("float32"),
    }


def save_prompt_bank(path: Path, bank: dict[str, np.ndarray | list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        class_ids=np.array(bank["class_ids"], dtype="int32"),
        class_names=np.array(bank["class_names"], dtype="U64"),
        means=np.array(bank["means"], dtype="float32"),
        stds=np.array(bank["stds"], dtype="float32"),
    )


def load_prompt_bank(path: Path) -> dict[str, np.ndarray | list[str]]:
    data = np.load(path, allow_pickle=False)
    return {
        "class_ids": data["class_ids"],
        "class_names": data["class_names"].tolist(),
        "means": data["means"],
        "stds": data["stds"],
    }


def select_class_index(prompt: str, class_names: list[str]) -> int:
    prompt_tokens = TOKEN_RE.findall(prompt.lower())
    joined = " ".join(prompt_tokens)

    exact_hits = [i for i, name in enumerate(class_names) if name.lower() in joined]
    if exact_hits:
        return exact_hits[0]

    name_tokens = [TOKEN_RE.findall(name.lower()) for name in class_names]
    token_scores = []
    for tokens in name_tokens:
        overlap = len(set(tokens).intersection(prompt_tokens))
        token_scores.append(overlap)

    best_idx = int(np.argmax(token_scores))
    if token_scores[best_idx] > 0:
        return best_idx

    close = difflib.get_close_matches(joined, [name.lower() for name in class_names], n=1, cutoff=0.5)
    if close:
        return [name.lower() for name in class_names].index(close[0])

    return 0


def sample_latents_for_prompt(
    prompt: str,
    bank: dict[str, np.ndarray | list[str]],
    num_images: int,
    seed: int | None = None,
) -> tuple[np.ndarray, str]:
    class_names = list(bank["class_names"])
    means = np.array(bank["means"])
    stds = np.array(bank["stds"])

    class_idx = select_class_index(prompt, class_names)
    rng = np.random.default_rng(seed)

    latents = means[class_idx] + rng.normal(size=(num_images, means.shape[1])) * stds[class_idx]
    return latents.astype("float32"), class_names[class_idx]
