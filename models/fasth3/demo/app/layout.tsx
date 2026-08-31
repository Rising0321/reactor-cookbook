import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "FastH3 live channel",
  description: "Queue prompts into a continuous FastH3 video and audio channel.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
