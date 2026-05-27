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
        display: ["'Exo 2'", "'Trebuchet MS'", "sans-serif"],
        body: ["'Merriweather Sans'", "'Segoe UI'", "sans-serif"],
        mono: ["'JetBrains Mono'", "'Consolas'", "monospace"],
      },
      boxShadow: {
        panel: "0 22px 40px -18px rgba(22, 39, 70, 0.45)",
        pulse: "0 0 0 1px rgba(53, 134, 255, 0.35), 0 0 34px rgba(53, 134, 255, 0.24)",
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
        reveal: "reveal 420ms cubic-bezier(0.23, 1, 0.32, 1) both",
        drift: "drift 5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
