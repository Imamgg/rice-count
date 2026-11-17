import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HitungBeras — Penghitung Butir Beras Otomatis (Computer Vision)",
  description:
    "Aplikasi penghitung jumlah butir beras berbasis Computer Vision langsung di browser tanpa perlu training model AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body className="min-h-screen antialiased selection:bg-emerald-500/30 selection:text-emerald-200">
        {children}
      </body>
    </html>
  );
}
