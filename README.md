# cnn-vae-2

Prompt-to-image app powered by your trained ConvVAE model.

Structure:
- `backend/`: model training, prompt bank generation, and FastAPI inference service
- `web/`: Next.js chat frontend

## 1) Train model + prompt bank

### Install backend deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r backend/requirements.txt
```

### Train/eval and build prompt bank (CIFAR-10)
```bash
python3 -m backend.train_eval \
  --dataset cifar10 \
  --train-split train \
  --eval-split test \
  --image-column img \
  --label-column label \
  --build-prompt-bank \
  --train-samples 5000 \
  --eval-samples 1000 \
  --epochs 5 \
  --batch-size 32 \
  --size 64 \
  --z-dim 32 \
  --beta 1.0
```

Artifacts in `artifacts/`:
- `vae.weights.h5`
- `prompt_bank.npz`
- `metrics.json`
- `generated_grid.png`
- `reconstruction_grid.png`

## 2) Run model API
```bash
source .venv/bin/activate
uvicorn backend.api_service:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `GET /health`
- `POST /generate` with JSON: `{ "prompt": "a frog", "num_images": 1, "seed": 42 }`

## 3) Run chat UI
```bash
cd web
npm install
cp .env.example .env.local
# MODEL_API_URL=http://localhost:8000
npm run dev
```

## Convenience commands
```bash
make backend-install
make backend-train
make backend-api
make web-install
make web-dev
```

## Deploy plan
- Frontend (`web/`) -> Vercel
- Python model API (`backend/api_service.py`) -> separate host (Render/Fly/Railway/VM)
- Optional containerization for backend via `backend/Dockerfile`

## Project files
- `backend/conv_vae.py`: VAE model
- `backend/data_hf.py`: dataset loading (images + labels)
- `backend/train_eval.py`: training/eval + prompt bank generation
- `backend/prompt_bank.py`: prompt-to-class latent sampling
- `backend/api_service.py`: FastAPI inference service
- `backend/.env.example`: backend artifact/env config
- `backend/Dockerfile`: backend container image
- `web/`: Next.js chat frontend

## License
MIT (see `LICENSE`).
