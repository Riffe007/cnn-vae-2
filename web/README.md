# Web App (Next.js)

Chat-style UI for prompt-driven generation from your trained ConvVAE backend.

## Local run
```bash
cd web
npm install
cp .env.example .env.local
# set MODEL_API_URL (default: http://localhost:8000)
npm run dev
```

## Vercel deployment
1. Import this repo into Vercel.
2. Set project root to `web`.
3. Add env var:
   - `MODEL_API_URL` (your deployed Python inference API)
4. Deploy.

## Important
Vercel hosts this frontend, but the Python/TensorFlow model API should run on a separate service (GPU/CPU host) and be referenced via `MODEL_API_URL`.
