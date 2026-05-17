import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AURA ONE — Spatial Dining Experience",
  description:
    "Gesture-controlled spatial restaurant experience powered by hand tracking and cinematic product presentation.",
};

// Safari iPad demo readiness:
// - viewportFit "cover" enables safe-area-inset-* for the notch / home indicator.
// - userScalable=false + maximumScale=1 prevents accidental pinch-zoom of the
//   page during demos (the canvas still handles its own interactions).
// - themeColor matches the stone-50 stage so the Safari address bar blends in.
export const viewport: Viewport = {
  themeColor: "#f6f2ea",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
