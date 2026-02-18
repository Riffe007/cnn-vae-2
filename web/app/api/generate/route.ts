import { NextRequest, NextResponse } from "next/server";

type GenerateBody = {
  prompt?: string;
  numImages?: number;
  seed?: number;
};

const MODEL_API_URL = process.env.MODEL_API_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = (await req.json()) as GenerateBody;
  const prompt = body.prompt?.trim();

  if (!prompt) {
    return NextResponse.json({ error: "Prompt is required" }, { status: 400 });
  }

  const response = await fetch(`${MODEL_API_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      num_images: clamp(body.numImages, 1, 8, 1),
      seed: body.seed,
    }),
    cache: "no-store",
  });

  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const err = (await response.json()) as { detail?: string; error?: string };
      detail = err.detail || err.error || detail;
    } catch {
      // ignored
    }
    return NextResponse.json({ error: `Model API failed: ${detail}` }, { status: 502 });
  }

  const payload = (await response.json()) as {
    matched_class: string;
    images_base64: string[];
  };

  return NextResponse.json({
    matchedClass: payload.matched_class,
    imagesBase64: payload.images_base64,
  });
}

function clamp(value: number | undefined, min: number, max: number, fallback: number) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, value));
}
