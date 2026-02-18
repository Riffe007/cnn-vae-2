# Backend

Python backend for training and serving ConvVAE prompt-to-image generation.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r backend/requirements.txt
```

## Train and create prompt bank
```bash
python3 -m backend.train_eval --dataset cifar10 --train-split train --eval-split test --image-column img --label-column label --build-prompt-bank
```

## Run API
```bash
uvicorn backend.api_service:app --host 0.0.0.0 --port 8000
```

## Env vars
Use `backend/.env.example` for model artifact paths.
