import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Fraunces'", "Georgia", "serif"],
        body: ["'IBM Plex Sans'", "'Segoe UI'", "sans-serif"],
        mono: ["'IBM Plex Mono'", "'Consolas'", "monospace"],
      },
      boxShadow: {
        panel: "0 28px 64px -28px rgba(5, 10, 18, 0.48)",
        pulse: "0 0 0 1px rgba(214, 155, 89, 0.32), 0 0 34px rgba(154, 168, 111, 0.22)",
      },
      keyframes: {
        reveal: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        drift: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-4px)" },
        },
      },
      animation: {
        reveal: "reveal 480ms cubic-bezier(0.23, 1, 0.32, 1) both",
        drift: "drift 6.5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
