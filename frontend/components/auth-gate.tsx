"use client";

import type { ReactNode, FormEvent } from "react";

import { AlertTriangle, LockKeyhole, LogIn, Loader2, Sparkles } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import {
  bootstrapDemoAuth,
  clearAuthToken,
  getStoredAuthToken,
  loginWithCredentials,
} from "@/lib/auth";

const DEFAULT_EMAIL = process.env.NEXT_PUBLIC_DEV_AUTH_EMAIL ?? "dev@helios.local";
const DEFAULT_PASSWORD = process.env.NEXT_PUBLIC_DEV_AUTH_PASSWORD ?? "dev-pass-123";

interface AuthGateProps {
  children: ReactNode;
}

export function AuthGate({ children }: AuthGateProps): JSX.Element {
  const [status, setStatus] = useState<"loading" | "anonymous" | "authenticated">("loading");
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
          setStatus("authenticated");
          return;
        }

        clearAuthToken();
      }

      if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
        const token = await bootstrapDemoAuth();
        if (!active) {
          return;
        }

        if (token) {
          setStatus("authenticated");
          return;
        }
      }

      setStatus("anonymous");
    };

    void initialize();

    return () => {
      active = false;
    };
  }, []);

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
      setStatus("authenticated");
    } catch (unknownError) {
      const message = unknownError instanceof Error ? unknownError.message : "Unable to sign in.";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const continueWithDemo = async () => {
    setIsSubmitting(true);
    setError(null);

    try {
      const token = await bootstrapDemoAuth();
      if (!token) {
        setError("Demo sign-in is unavailable right now. Please use the form below.");
        setStatus("anonymous");
        return;
      }
      setStatus("authenticated");
    } catch (unknownError) {
      const message = unknownError instanceof Error ? unknownError.message : "Unable to sign in.";
      setError(message);
      setStatus("anonymous");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (status === "authenticated") {
    return <>{children}</>;
  }

  if (status === "loading") {
    return (
      <main className="page-shell min-h-screen flex items-center px-4 py-4 lg:px-6 lg:py-6">
        <section className="panel-surface mx-auto flex flex-col min-h-0 w-full max-w-3xl items-center justify-center p-8 text-center">
          <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Authenticating
          </p>
          <h1 className="hero-display mt-4 text-4xl font-semibold tracking-[-0.03em] text-white md:text-5xl">
            Preparing Helios-Grid
          </h1>
          <p className="mt-2 text-sm text-slate-300 md:text-base">
            Checking your session and loading the demo account if needed.
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="page-shell min-h-screen flex items-center px-4 py-4 lg:px-6 lg:py-6">
      <div className="mx-auto grid min-h-0 max-w-7xl gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="panel-surface flex flex-col justify-between overflow-hidden p-6 lg:p-10">
          <div className="absolute right-0 top-0 h-64 w-64 -translate-y-1/2 translate-x-1/4 rounded-full bg-[radial-gradient(circle,rgba(214,155,89,0.18),transparent_68%)] blur-2xl" />
          <div className="relative z-10 max-w-2xl">
            <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">
              <LockKeyhole className="h-3.5 w-3.5" />
              Sign In Required
            </p>
            <h1 className="hero-display mt-5 text-4xl font-semibold tracking-[-0.03em] text-white md:text-6xl">
              Helios-Grid Mission Control
            </h1>
            <p className="mt-5 max-w-xl text-sm leading-7 text-slate-300 md:text-base">
              Sign in to continue. On localhost, the demo account can be created or logged in automatically.
            </p>
          </div>

          <div className="relative z-10 mt-10 grid gap-4 md:grid-cols-2">
            <article className="rounded-[1.15rem] border border-white/10 bg-white/5 px-4 py-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-[#f7ead1]">Verified session</p>
              <p className="mt-3 text-sm leading-6 text-slate-300">Existing tokens are checked before the workspace renders.</p>
            </article>
            <article className="rounded-[1.15rem] border border-white/10 bg-white/5 px-4 py-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-[#f7ead1]">Demo access</p>
              <p className="mt-3 text-sm leading-6 text-slate-300">Local development can bootstrap the demo account automatically.</p>
            </article>
          </div>
        </section>

        <section className="panel-surface flex items-center justify-center p-4 lg:p-6">
          <div className="w-full max-w-xl overflow-hidden rounded-[1.25rem] border border-white/10 bg-[rgba(10,15,22,0.58)] shadow-[0_20px_56px_rgba(5,10,18,0.22)]">
            <div className="border-b border-white/10 bg-white/[0.03] px-6 py-5">
              <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">
                <LockKeyhole className="h-3.5 w-3.5" />
                Sign In Required
              </p>
              <h2 className="hero-display mt-4 text-3xl font-semibold tracking-[-0.02em] text-white">
                Helios-Grid Mission Control
              </h2>
              <p className="mt-2 text-sm text-slate-300">
                Sign in to continue. On localhost, the demo account can be created or logged in automatically.
              </p>
            </div>

            <div className="space-y-5 px-6 py-6">
              {error ? (
                <div className="rounded-xl border border-rose-300/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
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
                    className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-slate-100 outline-none transition focus:border-[#d69b59]/60 focus:bg-white/[0.06]"
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
                    className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-slate-100 outline-none transition focus:border-[#d69b59]/60 focus:bg-white/[0.06]"
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
                    className="inline-flex items-center gap-2 rounded-full border border-[#d69b59]/35 bg-[#d69b59]/12 px-4 py-3 text-sm font-semibold text-[#f8e9cf] transition hover:bg-[#d69b59]/18 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogIn className="h-4 w-4" />}
                    Sign In
                  </button>

                  <button
                    type="button"
                    disabled={isSubmitting}
                    onClick={() => void continueWithDemo()}
                    className="inline-flex items-center gap-2 rounded-full border border-[#9aa86f]/35 bg-[#9aa86f]/12 px-4 py-3 text-sm font-semibold text-[#ebf0d6] transition hover:bg-[#9aa86f]/18 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                    Use Demo Account
                  </button>

                  <Link
                    href="/login"
                    className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-200 transition hover:border-white/20 hover:bg-white/[0.07]"
                  >
                    Separate Login Page
                  </Link>
                </div>
              </form>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
