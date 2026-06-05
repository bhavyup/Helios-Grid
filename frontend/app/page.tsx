import Link from "next/link";
import { ArrowRight, BarChart3, Globe2, LockKeyhole, Shield, Sparkles } from "lucide-react";

export default function HomePage(): JSX.Element {
  return (
    <main className="page-shell min-h-screen px-4 py-4 lg:px-6 lg:py-6">
      <div className="mx-auto grid min-h-[calc(100vh-2rem)] w-full max-w-7xl gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="panel-surface relative overflow-hidden p-6 lg:p-10">
          <div className="absolute inset-0 surface-grid" />
          <div className="absolute right-0 top-0 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgba(212,175,55,0.14),transparent_68%)] blur-3xl" />
          <div className="relative z-10 flex h-full flex-col justify-between gap-10">
            <div className="max-w-3xl">
              <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em]">
                <Sparkles className="h-3.5 w-3.5" />
                Helios-Grid / Mission Control
              </p>
              <h1 className="hero-display mt-5 max-w-4xl text-5xl font-semibold tracking-[-0.04em] text-white md:text-7xl">
                A quieter, sharper operating surface for neighborhood energy.
              </h1>
              <p className="section-copy mt-5 max-w-2xl text-base md:text-lg">
                Explore simulation, weather ingestion, and PPO analytics in a workspace designed like a modern control room rather than a cluttered dashboard.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Link
                  href="/login"
                  className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-5 py-3 text-sm font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)]"
                >
                  <LockKeyhole className="h-4 w-4" />
                  Sign in
                </Link>
                <Link
                  href="/dashboard"
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-white/20 hover:bg-white/[0.07]"
                >
                  Open dashboard
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              {[
                ["Weather-first", "Upload, profile, derive, and reset from one focused workflow."],
                ["Grid clarity", "See topology, state, and flows without reading a data dump."],
                ["Training lens", "Compare PPO and rule-based behavior in one visual frame."],
              ].map(([title, description]) => (
                <article key={title} className="panel-frame rounded-[1.2rem] px-4 py-4">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-[#f6e7be]">{title}</p>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{description}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <aside className="space-y-6">
          <section className="panel-surface overflow-hidden p-6 lg:p-8">
            <p className="section-eyebrow">Operational stack</p>
            <h2 className="hero-display mt-3 text-3xl font-semibold tracking-[-0.03em] text-white">
              Built as a dashboard system, not a page dump.
            </h2>
            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              {[
                [BarChart3, "Analytics", "Reward curves, baseline gap, and exportable artifacts."],
                [Globe2, "Grid state", "Topology, weather source, and live simulation controls."],
                [Shield, "Access", "Login-first flow with protected runtime actions."],
                [Sparkles, "Motion", "Soft glow, glass panels, and editorial rhythm."],
              ].map(([Icon, title, description]) => (
                <article key={title as string} className="panel-frame rounded-[1.2rem] p-4">
                  <Icon className="h-5 w-5 text-[#d4af37]" />
                  <p className="mt-4 text-sm font-semibold text-white">{title as string}</p>
                  <p className="mt-2 text-sm leading-6 text-slate-300">{description as string}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel-surface overflow-hidden p-6 lg:p-8">
            <p className="section-eyebrow">Quick entry</p>
            <h3 className="hero-display mt-3 text-2xl font-semibold tracking-[-0.03em] text-white">
              Jump straight into the mission room.
            </h3>
            <div className="mt-5 flex flex-wrap gap-3">
              <Link
                href="/login"
                className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
              >
                Login
              </Link>
              <Link
                href="/dashboard"
                className="rounded-full border border-[rgba(127,182,168,0.3)] bg-[rgba(127,182,168,0.12)] px-4 py-3 text-sm font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)]"
              >
                Dashboard
              </Link>
            </div>
          </section>
        </aside>
      </div>
    </main>
  );
}
