"""Train and evaluate ConvVAE on Hugging Face datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from backend.conv_vae import ConvVAE
from backend.data_hf import load_hf_images, load_hf_images_and_labels
from backend.prompt_bank import build_prompt_bank, save_prompt_bank


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def reconstruction_mse(model: ConvVAE, x: np.ndarray) -> float:
    recon = model.reconstruct(x)
    return float(np.mean(np.square(x - recon)))


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _save_grid(images: np.ndarray, path: Path, title: str, cols: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < len(images):
            ax.imshow(np.clip(images[i], 0.0, 1.0))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_reconstruction_grid(model: ConvVAE, x: np.ndarray, path: Path, max_items: int = 8) -> None:
    sample = x[:max_items]
    recon = model.reconstruct(sample)
    merged = np.concatenate([sample, recon], axis=0)
    _save_grid(merged, path, title="Top: Original | Bottom: Reconstruction", cols=max_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate CNN-VAE with Hugging Face datasets")
    parser.add_argument("--dataset", default="cifar10", help="HF dataset id, e.g. cifar10")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--image-column", default="img", help="Image column name (cifar10 uses 'img')")
    parser.add_argument("--label-column", default="label", help="Label column used for prompt bank generation")
    parser.add_argument("--build-prompt-bank", action="store_true", help="Build class-conditioned prompt bank")
    parser.add_argument("--size", type=int, default=64, help="Square image size")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv-filters", default="32,64,64,128")
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-generated-samples", type=int, default=16)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    conv_filters = parse_int_list(args.conv_filters)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    class_names: list[str] = []
    if args.build_prompt_bank:
        x_train, labels, class_names = load_hf_images_and_labels(
            dataset_name=args.dataset,
            split=args.train_split,
            image_column=args.image_column,
            label_column=args.label_column,
            image_size=(args.size, args.size),
            max_samples=args.train_samples,
        )
    else:
        x_train = load_hf_images(
            dataset_name=args.dataset,
            split=args.train_split,
            image_column=args.image_column,
            image_size=(args.size, args.size),
            max_samples=args.train_samples,
        )

    x_eval = load_hf_images(
        dataset_name=args.dataset,
        split=args.eval_split,
        image_column=args.image_column,
        image_size=(args.size, args.size),
        max_samples=args.eval_samples,
    )

    vae = ConvVAE(
        input_dim=(args.size, args.size, 3),
        z_dim=args.z_dim,
        conv_filters=conv_filters,
        kernel_size=args.kernel_size,
        stride=args.stride,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
    )
    history = vae.train(x_train, epochs=args.epochs, batch_size=args.batch_size)

    train_mse = reconstruction_mse(vae, x_train[: min(512, len(x_train))])
    eval_mse = reconstruction_mse(vae, x_eval)

    generated = vae.generate(args.num_generated_samples, seed=args.seed)

    weights_path = out_dir / "vae.weights.h5"
    metrics_path = out_dir / "metrics.json"
    generated_grid_path = out_dir / "generated_grid.png"
    reconstruction_grid_path = out_dir / "reconstruction_grid.png"
    prompt_bank_path = out_dir / "prompt_bank.npz"

    vae.save_weights(str(weights_path))
    _save_grid(generated, generated_grid_path, title="Random samples")
    _save_reconstruction_grid(vae, x_eval, reconstruction_grid_path)

    if args.build_prompt_bank:
        z_mean, _, _ = vae.encoder.predict(x_train, verbose=0)
        bank = build_prompt_bank(z_mean=z_mean, labels=np.array(labels), class_names=class_names)
        save_prompt_bank(prompt_bank_path, bank)

    save_metrics(
        metrics_path,
        {
            "dataset": args.dataset,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "image_column": args.image_column,
            "label_column": args.label_column if args.build_prompt_bank else None,
            "build_prompt_bank": args.build_prompt_bank,
            "image_size": args.size,
            "train_samples": len(x_train),
            "eval_samples": len(x_eval),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "z_dim": args.z_dim,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "seed": args.seed,
            "conv_filters": conv_filters,
            "kernel_size": args.kernel_size,
            "stride": args.stride,
            "train_reconstruction_mse": train_mse,
            "eval_reconstruction_mse": eval_mse,
            "best_val_loss": min(history.history.get("val_loss", [float("nan")])) if history.history else float("nan"),
            "weights_path": str(weights_path),
            "generated_grid_path": str(generated_grid_path),
            "reconstruction_grid_path": str(reconstruction_grid_path),
            "prompt_bank_path": str(prompt_bank_path) if args.build_prompt_bank else None,
        },
    )

    print(f"Saved weights to: {weights_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved generated grid to: {generated_grid_path}")
    print(f"Saved reconstruction grid to: {reconstruction_grid_path}")
    if args.build_prompt_bank:
        print(f"Saved prompt bank to: {prompt_bank_path}")
    print(f"Train reconstruction MSE: {train_mse:.6f}")
    print(f"Eval reconstruction MSE: {eval_mse:.6f}")


if __name__ == "__main__":
    main()
