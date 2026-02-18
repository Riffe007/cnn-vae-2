.PHONY: backend-install backend-train backend-api web-install web-dev

backend-install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r backend/requirements.txt

backend-train:
	. .venv/bin/activate && python3 -m backend.train_eval --dataset cifar10 --train-split train --eval-split test --image-column img --label-column label --build-prompt-bank --train-samples 5000 --eval-samples 1000 --epochs 5 --batch-size 32 --size 64 --z-dim 32 --beta 1.0

backend-api:
	. .venv/bin/activate && uvicorn backend.api_service:app --host 0.0.0.0 --port 8000

web-install:
	cd web && npm install

web-dev:
	cd web && npm run dev
