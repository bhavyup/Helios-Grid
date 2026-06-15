"use client";

import { AlertTriangle, ArrowRight, LockKeyhole, Loader2, Sparkles } from "lucide-react";
import type { FormEvent } from "react";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { clearAuthToken, getStoredAuthToken, loginWithCredentials } from "@/lib/auth";

const DEFAULT_EMAIL = process.env.NEXT_PUBLIC_DEV_AUTH_EMAIL ?? "dev@helios.local";
const DEFAULT_PASSWORD = process.env.NEXT_PUBLIC_DEV_AUTH_PASSWORD ?? "dev-pass-123";

export default function LoginPage(): JSX.Element {
  const router = useRouter();
  const [email, setEmail] = useState(DEFAULT_EMAIL);
  const [password, setPassword] = useState(DEFAULT_PASSWORD);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    let active = true;

    const verifyToken = async (token: string) => {
      const response = await fetch("/api/backend/auth/me", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      return response.ok;
    };

    const initialize = async () => {
      const existing = getStoredAuthToken();
      if (existing) {
        const valid = await verifyToken(existing);
        if (!active) {
          return;
        }

        if (valid) {
          router.replace("/dashboard");
          return;
        }

        clearAuthToken();
      }
    };

    void initialize();

    return () => {
      active = false;
    };
  }, [router]);

  const submitLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const token = await loginWithCredentials({ email, password });
      if (!token) {
        setError("Login failed. Check your email and password, then try again.");
        return;
      }
      router.replace("/dashboard");
    } catch (unknownError) {
      const message = unknownError instanceof Error ? unknownError.message : "Unable to sign in.";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="page-shell min-h-screen px-4 py-4 lg:px-6 lg:py-6">
      <div className="mx-auto grid min-h-[calc(100vh-2rem)] max-w-7xl gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <section className="panel-surface relative overflow-hidden p-6 lg:p-10">
          <div className="absolute inset-0 surface-grid" />
          <div className="absolute right-0 top-0 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgba(212,175,55,0.15),transparent_68%)] blur-3xl" />
          <div className="absolute left-0 top-20 h-44 w-44 rounded-full bg-[radial-gradient(circle,rgba(127,182,168,0.14),transparent_72%)] blur-3xl" />

          <div className="relative z-10 flex h-full flex-col justify-between gap-10">
            <div className="max-w-3xl">
              <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em]">
                <LockKeyhole className="h-3.5 w-3.5" />
                Access console
              </p>
              <h1 className="hero-display mt-5 max-w-3xl text-5xl font-semibold tracking-[-0.04em] text-white md:text-7xl">
                Mission control, not another login wall.
              </h1>
              <p className="section-copy mt-5 max-w-2xl text-base md:text-lg">
                Sign in to enter a premium simulation workspace where weather ingestion, grid control, and PPO analytics sit inside one calm, highly readable interface.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              {[
                ["Protected access", "Existing tokens are verified before the dashboard renders."],
                ["Demo path", "Local development can still bootstrap the demo account fast."],
                ["One workspace", "Weather, topology, training, and exports all live together."],
              ].map(([title, description]) => (
                <article key={title} className="panel-frame rounded-[1.2rem] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-[#f6e7be]">{title}</p>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{description}</p>
                </article>
              ))}
            </div>

            <div className="flex flex-wrap gap-3 text-[11px] uppercase tracking-[0.18em] text-slate-300">
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2">Decentralized energy</span>
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2">Premium glass shell</span>
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2">Direct dashboard access</span>
            </div>
          </div>
        </section>

        <section className="panel-surface flex items-center justify-center p-4 lg:p-6">
          <div className="w-full max-w-xl overflow-hidden rounded-[1.4rem] border border-white/10 bg-[rgba(8,10,14,0.62)] shadow-[0_24px_60px_rgba(0,0,0,0.34)]">
            <div className="border-b border-white/10 bg-white/[0.03] px-6 py-6">
              <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em]">
                <Sparkles className="h-3.5 w-3.5" />
                Sign in
              </p>
              <h2 className="hero-display mt-4 text-3xl font-semibold tracking-[-0.03em] text-white">
                Continue into the mission room.
              </h2>
              <p className="section-copy mt-2 text-sm">
                Use your account or the demo account to continue into the dashboard.
              </p>
            </div>

            <div className="space-y-5 px-6 py-6">
              {error ? (
                <div className="rounded-[1.1rem] border border-rose-300/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  <p className="inline-flex items-center gap-2 font-semibold uppercase tracking-[0.12em]">
                    <AlertTriangle className="h-4 w-4" />
                    Login Error
                  </p>
                  <p className="mt-1 text-rose-200">{error}</p>
                </div>
              ) : null}

              <form className="grid gap-4" onSubmit={submitLogin}>
                <label className="grid gap-2 text-sm text-slate-200">
                  Email
                  <input
                    className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.55)] focus:bg-white/[0.06]"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    type="email"
                    autoComplete="email"
                    placeholder="dev@helios.local"
                  />
                </label>

                <label className="grid gap-2 text-sm text-slate-200">
                  Password
                  <input
                    className="rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.55)] focus:bg-white/[0.06]"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    type="password"
                    autoComplete="current-password"
                    placeholder="dev-pass-123"
                  />
                </label>

                <div className="flex flex-wrap gap-3 pt-2">
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-4 py-3 text-sm font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <LockKeyhole className="h-4 w-4" />}
                    Sign In
                  </button>

                  <button
                    type="button"
                    disabled={isSubmitting}
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-60"
                    onClick={() => setEmail(DEFAULT_EMAIL)}
                  >
                    <ArrowRight className="h-4 w-4" />
                    Reset email
                  </button>
                </div>

                <p className="pt-2 text-xs text-slate-400">
                  Sign in with your email and password to continue.
                </p>
              </form>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
