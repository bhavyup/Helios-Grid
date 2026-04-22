"use client";

import { BarChart3, Brain, GitCompare, Loader2 } from "lucide-react";
import { useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PPOComparisonInput, PPOTrainingInput } from "@/lib/api-client";
import { PolicyComparison, RewardCurvePayload, TrainingRunPayload } from "@/lib/types";

interface TrainingPanelProps {
  latestRun: TrainingRunPayload | null;
  latestComparison: (PolicyComparison & { run_id?: string; created_at?: string }) | null;
  rewardCurve: RewardCurvePayload | null;
  isTraining: boolean;
  error: string | null;
  onTrain: (input: PPOTrainingInput) => Promise<void>;
  onCompare: (input: PPOComparisonInput) => Promise<void>;
  onRefresh: () => Promise<void>;
}

function deltaTone(value: number): string {
  if (value > 0) {
    return "text-emerald-300";
  }
  if (value < 0) {
    return "text-rose-300";
  }
  return "text-slate-300";
}

export function TrainingPanel({
  latestRun,
  latestComparison,
  rewardCurve,
  isTraining,
  error,
  onTrain,
  onCompare,
  onRefresh,
}: TrainingPanelProps): JSX.Element {
  const [episodes, setEpisodes] = useState<number>(20);
  const [stepsPerEpisode, setStepsPerEpisode] = useState<number>(24);
  const [evalEpisodes, setEvalEpisodes] = useState<number>(5);

  const handleTrain = async () => {
    await onTrain({
      episodes,
      steps_per_episode: stepsPerEpisode,
      eval_episodes: evalEpisodes,
    });
  };

  const handleCompare = async () => {
    await onCompare({
      episodes: Math.max(2, evalEpisodes),
      steps_per_episode: stepsPerEpisode,
    });
  };

  const comparison = latestComparison ?? latestRun?.comparison ?? null;
  const curve = rewardCurve?.reward_curve ?? latestRun?.training.reward_curve ?? [];

  return (
    <section className="panel-surface h-full px-5 py-5">
      <div className="mb-3 flex items-start justify-between">
        <div>
          <h2 className="panel-title">PPO Progress & Baseline Gap</h2>
          <p className="mt-1 text-xs text-slate-400">
            Train PPO proof-of-life, then inspect reward trajectory and rule-vs-PPO deltas.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void onRefresh()}
          className="rounded-md border border-slate-700 bg-slate-900/70 px-3 py-1 text-xs font-semibold text-slate-300 transition hover:border-cyan-400/70 hover:text-cyan-200"
        >
          Refresh Artifacts
        </button>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <label className="space-y-1 text-xs text-slate-300">
          <span className="uppercase tracking-[0.15em] text-slate-400">Episodes</span>
          <input
            type="number"
            min={1}
            max={500}
            value={episodes}
            onChange={(event) => setEpisodes(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
          />
        </label>
        <label className="space-y-1 text-xs text-slate-300">
          <span className="uppercase tracking-[0.15em] text-slate-400">Steps / Episode</span>
          <input
            type="number"
            min={1}
            max={500}
            value={stepsPerEpisode}
            onChange={(event) => setStepsPerEpisode(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
          />
        </label>
        <label className="space-y-1 text-xs text-slate-300">
          <span className="uppercase tracking-[0.15em] text-slate-400">Eval Episodes</span>
          <input
            type="number"
            min={1}
            max={100}
            value={evalEpisodes}
            onChange={(event) => setEvalEpisodes(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
          />
        </label>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <button
          type="button"
          disabled={isTraining}
          onClick={() => void handleTrain()}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-indigo-300/70 bg-indigo-500/15 px-3 py-2 text-sm font-semibold text-indigo-100 transition hover:bg-indigo-500/25 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
          {isTraining ? "Training..." : "Run PPO Training"}
        </button>
        <button
          type="button"
          disabled={isTraining}
          onClick={() => void handleCompare()}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-emerald-300/70 bg-emerald-500/15 px-3 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <GitCompare className="h-4 w-4" />
          Refresh PPO vs Rule
        </button>
      </div>

      {error ? (
        <p className="mt-3 rounded-md border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">{error}</p>
      ) : null}

      <div className="mt-4 rounded-xl border border-slate-700/70 bg-slate-950/60 p-3">
        <div className="mb-2 flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-300">
          <BarChart3 className="h-3.5 w-3.5" />
          Reward Curve
        </div>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={curve}>
              <XAxis dataKey="episode" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "rgba(15, 23, 42, 0.95)",
                  border: "1px solid rgba(100, 116, 139, 0.5)",
                  borderRadius: 8,
                }}
              />
              <Line type="monotone" dataKey="reward" stroke="#38bdf8" strokeWidth={2.2} dot={false} />
              <Line
                type="monotone"
                dataKey="moving_average_reward"
                stroke="#f59e0b"
                strokeWidth={2.2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <article className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">PPO Avg Reward</p>
          <p className="mt-1 font-display text-xl text-white">
            {(comparison?.ppo.average_reward ?? latestRun?.training.final_eval_metrics.average_reward ?? 0).toFixed(3)}
          </p>
        </article>
        <article className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Rule Avg Reward</p>
          <p className="mt-1 font-display text-xl text-white">{(comparison?.rule.average_reward ?? 0).toFixed(3)}</p>
        </article>
        <article className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Reward Delta</p>
          <p className={`mt-1 font-display text-xl ${deltaTone(comparison?.deltas.reward_delta ?? 0)}`}>
            {(comparison?.deltas.reward_delta ?? 0).toFixed(3)}
          </p>
        </article>
      </div>
    </section>
  );
}
