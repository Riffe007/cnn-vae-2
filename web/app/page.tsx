"use client";

import { useState } from "react";

type AssistantMessage = {
  prompt: string;
  matchedClass: string;
  imagesBase64: string[];
};

export default function Home() {
  const [prompt, setPrompt] = useState("a frog in nature");
  const [numImages, setNumImages] = useState(1);
  const [seed, setSeed] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<AssistantMessage[]>([]);

  async function onSend() {
    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          numImages,
          seed: seed.trim() ? Number(seed) : undefined,
        }),
      });

      if (!res.ok) {
        const payload = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(payload.error || `Request failed with status ${res.status}`);
      }

      const data = (await res.json()) as { matchedClass: string; imagesBase64: string[] };
      setMessages((prev) => [{ prompt, matchedClass: data.matchedClass, imagesBase64: data.imagesBase64 }, ...prev]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>VAE Prompt Studio</h1>
      <p className="meta">Prompt -> class-conditioned latent sampling -> ConvVAE decode</p>

      <section className="panel row">
        <label>
          Prompt
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        </label>

        <div className="controls">
          <label>
            Images
            <select value={numImages} onChange={(e) => setNumImages(Number(e.target.value))}>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={4}>4</option>
            </select>
          </label>
          <label>
            Seed (optional)
            <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="42" />
          </label>
        </div>

        <button disabled={loading || !prompt.trim()} onClick={onSend}>
          {loading ? "Generating..." : "Send Prompt"}
        </button>

        {error ? <p>{error}</p> : null}
      </section>

      <section className="preview">
        {messages.map((msg, idx) => (
          <article className="panel" key={`${msg.prompt}-${idx}`}>
            <p>
              <strong>Prompt:</strong> {msg.prompt}
            </p>
            <p className="meta">
              <strong>Matched class:</strong> {msg.matchedClass}
            </p>
            <div className="gallery">
              {msg.imagesBase64.map((img, imgIdx) => (
                <img key={`${idx}-${imgIdx}`} src={`data:image/png;base64,${img}`} alt={msg.prompt} />
              ))}
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
