import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Echo-WM Flash — audiovisual world model",
  description:
    "Drive an audiovisual world in real time: pick a starting image, write a prompt, and steer four camera axes from the keyboard.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
