# cnn-vae-2

Prompt-to-image app powered by your trained ConvVAE model.

Architecture:
- Python training/eval pipeline builds model weights and a prompt bank.
- Python FastAPI service loads those artifacts and returns generated images from prompt text.
- Next.js chat UI calls the FastAPI service and displays generated results.

## 1) Train model + prompt bank

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Train/eval and build prompt bank (CIFAR-10)
```bash
python3 train_eval.py \
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
uvicorn api_service:app --host 0.0.0.0 --port 8000
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

## Deploy plan
- Frontend (`web/`) -> Vercel
- Python model API (`api_service.py`) -> separate host (Render/Fly/Railway/VM)

## Dataset guidance
For this class-conditioned prompt flow, use labeled datasets:
- `cifar10`
- `fashion_mnist`
- `beans`

## Project files
- `conv_vae.py`: VAE model
- `data_hf.py`: dataset loading (images + labels)
- `train_eval.py`: training/eval + prompt bank generation
- `prompt_bank.py`: prompt-to-class latent sampling logic
- `api_service.py`: FastAPI inference service
- `web/`: Next.js chat frontend

## License
MIT (see `LICENSE`).
