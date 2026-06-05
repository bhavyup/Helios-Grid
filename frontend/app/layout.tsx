import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Helios-Grid Mission Control",
  description:
    "Premium decentralized energy simulation workspace for weather ingestion, live controls, PPO training, and grid analytics.",
};

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps): JSX.Element {
  return (
    <html lang="en">
      <body className="bg-[var(--bg-0)] font-body antialiased text-white">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
