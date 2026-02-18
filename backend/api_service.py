"""FastAPI service for prompt-to-image generation using trained ConvVAE."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

from backend.conv_vae import ConvVAE
from backend.prompt_bank import load_prompt_bank, sample_latents_for_prompt


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=400)
    num_images: int = Field(default=1, ge=1, le=8)
    seed: int | None = None


class GenerateResponse(BaseModel):
    prompt: str
    matched_class: str
    images_base64: list[str]


def _img_to_base64(img: np.ndarray) -> str:
    arr = np.clip(img * 255.0, 0.0, 255.0).astype("uint8")
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_model_from_metrics(metrics_path: Path) -> ConvVAE:
    import json

    if not metrics_path.exists():
        return ConvVAE()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    size = int(metrics.get("image_size", 64))
    z_dim = int(metrics.get("z_dim", 32))
    conv_filters = tuple(int(v) for v in metrics.get("conv_filters", [32, 64, 64, 128]))
    kernel_size = int(metrics.get("kernel_size", 4))
    stride = int(metrics.get("stride", 2))
    learning_rate = float(metrics.get("learning_rate", 1e-3))
    beta = float(metrics.get("beta", 1.0))

    return ConvVAE(
        input_dim=(size, size, 3),
        z_dim=z_dim,
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        stride=stride,
        learning_rate=learning_rate,
        beta=beta,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="cnn-vae prompt inference", version="0.1.0")

    weights_path = Path(os.getenv("MODEL_WEIGHTS", "artifacts/vae.weights.h5"))
    prompt_bank_path = Path(os.getenv("PROMPT_BANK_PATH", "artifacts/prompt_bank.npz"))
    metrics_path = Path(os.getenv("MODEL_METRICS", "artifacts/metrics.json"))

    if not weights_path.exists():
        raise RuntimeError(f"Model weights not found: {weights_path}")
    if not prompt_bank_path.exists():
        raise RuntimeError(f"Prompt bank not found: {prompt_bank_path}. Train with --build-prompt-bank")

    model = _load_model_from_metrics(metrics_path)
    model.load_weights(str(weights_path))
    bank = load_prompt_bank(prompt_bank_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        prompt = req.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        latents, matched_class = sample_latents_for_prompt(
            prompt=prompt,
            bank=bank,
            num_images=req.num_images,
            seed=req.seed,
        )
        images = model.decode_latents(latents)
        images_base64 = [_img_to_base64(img) for img in images]

        return GenerateResponse(
            prompt=prompt,
            matched_class=matched_class,
            images_base64=images_base64,
        )

    return app


app = create_app()
