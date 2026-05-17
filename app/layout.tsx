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
  applicationName: "AURA ONE",
  title: "AURA ONE — Spatial Dining Experience",
  description:
    "Gesture-controlled spatial restaurant experience powered by hand tracking and cinematic product presentation.",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "AURA ONE",
  },
  formatDetection: {
    telephone: false,
  },
  icons: {
    icon: [
      { url: "/icons/aura-icon.svg", type: "image/svg+xml" },
    ],
    apple: [
      { url: "/icons/aura-apple-icon.svg", type: "image/svg+xml" },
    ],
  },
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
      className={`${geistSans.variable} ${geistMono.variable} h-full overflow-hidden overscroll-none antialiased`}
    >
      <body className="h-full overflow-hidden overscroll-none bg-black">{children}</body>
    </html>
  );
}
