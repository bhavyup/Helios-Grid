"use client";

import { Download } from "lucide-react";

import {
  PolicyComparison,
  RewardCurvePayload,
  SimulationMetrics,
  TrajectoryPoint,
  TrainingRunPayload,
} from "@/lib/types";

interface ExportArtifactsProps {
  metrics: SimulationMetrics | null;
  history: TrajectoryPoint[];
  latestRun: TrainingRunPayload | null;
  latestComparison: (PolicyComparison & { run_id?: string; created_at?: string }) | null;
  rewardCurve: RewardCurvePayload | null;
}

function downloadFile(fileName: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const objectUrl = URL.createObjectURL(blob);

  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = fileName;
  anchor.click();

  URL.revokeObjectURL(objectUrl);
}

function buildHistoryCsv(history: TrajectoryPoint[]): string {
  if (history.length === 0) {
    return "step,timestamp,reward,done,supply,demand,price,grid_import,renewable_utilization\n";
  }

  const header = [
    "step",
    "timestamp",
    "reward",
    "done",
    "supply",
    "demand",
    "price",
    "grid_import",
    "renewable_utilization",
  ];

  const rows = history.map((point) => [
    point.step,
    point.timestamp,
    point.reward,
    point.done,
    point.supply,
    point.demand,
    point.price,
    point.grid_import,
    point.renewable_utilization,
  ]);

  return [header.join(","), ...rows.map((row) => row.join(","))].join("\n");
}

export function ExportArtifacts({
  metrics,
  history,
  latestRun,
  latestComparison,
  rewardCurve,
}: ExportArtifactsProps): JSX.Element {
  const hasSimulationData = history.length > 0 || metrics !== null;
  const hasTrainingData = latestRun !== null || latestComparison !== null || (rewardCurve?.reward_curve.length ?? 0) > 0;

  const exportSimulationCsv = () => {
    const csv = buildHistoryCsv(history);
    const fileName = `helios-simulation-history-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    downloadFile(fileName, csv, "text/csv;charset=utf-8");
  };

  const exportMetricsJson = () => {
    const payload = {
      exported_at: new Date().toISOString(),
      metrics,
      history_points: history.length,
    };
    const fileName = `helios-simulation-metrics-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    downloadFile(fileName, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
  };

  const exportTrainingJson = () => {
    const payload = {
      exported_at: new Date().toISOString(),
      latest_run: latestRun,
      comparison: latestComparison,
      reward_curve: rewardCurve,
    };
    const fileName = `helios-training-artifacts-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    downloadFile(fileName, JSON.stringify(payload, null, 2), "application/json;charset=utf-8");
  };

  return (
    <section className="panel-surface mt-4 px-5 py-5 lg:px-6 lg:py-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="section-eyebrow">Artifact export</p>
          <h2 className="hero-display mt-2 text-3xl font-semibold tracking-[-0.03em] text-white">Export the story, not just the numbers.</h2>
          <p className="section-copy mt-2 text-sm">
            Download chart-ready simulation and training outputs for faculty review, sharing, or external analysis.
          </p>
        </div>

        <div className="grid gap-2 sm:grid-cols-3">
          <button
            type="button"
            disabled={!hasSimulationData}
            onClick={exportSimulationCsv}
            className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-3 text-xs font-semibold uppercase tracking-[0.1em] text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5" />
            Simulation CSV
          </button>

          <button
            type="button"
            disabled={!hasSimulationData}
            onClick={exportMetricsJson}
            className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-3 text-xs font-semibold uppercase tracking-[0.1em] text-slate-100 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5" />
            Metrics JSON
          </button>

          <button
            type="button"
            disabled={!hasTrainingData}
            onClick={exportTrainingJson}
            className="inline-flex items-center justify-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-3 py-3 text-xs font-semibold uppercase tracking-[0.1em] text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)] disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Download className="h-3.5 w-3.5" />
            Training JSON
          </button>
        </div>
      </div>
    </section>
  );
}
