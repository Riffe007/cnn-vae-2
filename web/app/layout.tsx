import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Text to Image Lab",
  description: "Generate images with hosted diffusion models",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
