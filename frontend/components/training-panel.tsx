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
    <section className="panel-surface h-full px-5 py-5 lg:px-6 lg:py-6">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="section-eyebrow">Learning stack</p>
          <h2 className="hero-display mt-2 text-3xl font-semibold tracking-[-0.03em] text-white">PPO progress with a quiet baseline gap readout.</h2>
          <p className="section-copy mt-2 text-sm">Train PPO proof-of-life, then inspect reward trajectory and rule-vs-PPO deltas.</p>
        </div>
        <button
          type="button"
          onClick={() => void onRefresh()}
          className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-semibold text-slate-300 transition hover:border-[rgba(212,175,55,0.4)] hover:text-[#f6e7be]"
        >
          Refresh Artifacts
        </button>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <label className="space-y-1 text-xs text-slate-300">
          <span className="section-eyebrow text-[10px]">Episodes</span>
          <input
            type="number"
            min={1}
            max={500}
            value={episodes}
            onChange={(event) => setEpisodes(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-3 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
          />
        </label>
        <label className="space-y-1 text-xs text-slate-300">
          <span className="section-eyebrow text-[10px]">Steps / episode</span>
          <input
            type="number"
            min={1}
            max={500}
            value={stepsPerEpisode}
            onChange={(event) => setStepsPerEpisode(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-3 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
          />
        </label>
        <label className="space-y-1 text-xs text-slate-300">
          <span className="section-eyebrow text-[10px]">Eval episodes</span>
          <input
            type="number"
            min={1}
            max={100}
            value={evalEpisodes}
            onChange={(event) => setEvalEpisodes(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-2xl border border-white/10 bg-[rgba(255,255,255,0.03)] px-3 py-3 text-sm text-slate-100 outline-none transition focus:border-[rgba(212,175,55,0.45)]"
          />
        </label>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <button
          type="button"
          disabled={isTraining}
          onClick={() => void handleTrain()}
          className="inline-flex items-center justify-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-3 py-3 text-sm font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isTraining ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
          {isTraining ? "Training..." : "Run PPO Training"}
        </button>
        <button
          type="button"
          disabled={isTraining}
          onClick={() => void handleCompare()}
          className="inline-flex items-center justify-center gap-2 rounded-full border border-[rgba(127,182,168,0.34)] bg-[rgba(127,182,168,0.12)] px-3 py-3 text-sm font-semibold text-[#d8f2eb] transition hover:bg-[rgba(127,182,168,0.18)] disabled:cursor-not-allowed disabled:opacity-60"
        >
          <GitCompare className="h-4 w-4" />
          Refresh PPO vs Rule
        </button>
      </div>

      {error ? (
        <p className="mt-3 rounded-[1rem] border border-rose-400/30 bg-rose-500/10 px-3 py-3 text-xs text-rose-200">{error}</p>
      ) : null}

      <div className="mt-4 rounded-[1.35rem] border border-white/10 bg-white/[0.03] p-3 lg:p-4">
        <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-slate-300">
          <BarChart3 className="h-3.5 w-3.5" />
          Reward Curve
        </div>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={curve}>
              <XAxis dataKey="episode" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#a8b0bc" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: "rgba(9, 12, 17, 0.97)",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  borderRadius: 16,
                }}
              />
              <Line type="monotone" dataKey="reward" stroke="#f0c46b" strokeWidth={2.2} dot={false} />
              <Line
                type="monotone"
                dataKey="moving_average_reward"
                stroke="#7fb6a8"
                strokeWidth={2.2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <article className="panel-frame rounded-[1rem] px-3 py-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">PPO Avg Reward</p>
          <p className="hero-display mt-1 text-xl text-white">
            {(comparison?.ppo.average_reward ?? latestRun?.training.final_eval_metrics.average_reward ?? 0).toFixed(3)}
          </p>
        </article>
        <article className="panel-frame rounded-[1rem] px-3 py-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Rule Avg Reward</p>
          <p className="hero-display mt-1 text-xl text-white">{(comparison?.rule.average_reward ?? 0).toFixed(3)}</p>
        </article>
        <article className="panel-frame rounded-[1rem] px-3 py-3">
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">Reward Delta</p>
          <p className={`hero-display mt-1 text-xl ${deltaTone(comparison?.deltas.reward_delta ?? 0)}`}>
            {(comparison?.deltas.reward_delta ?? 0).toFixed(3)}
          </p>
        </article>
      </div>
    </section>
  );
}
