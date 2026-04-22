"use client";

import { AlertTriangle, Bot, Cpu, Sparkles } from "lucide-react";
import { useState } from "react";

import { EnergyCharts } from "@/components/energy-charts";
import { ExportArtifacts } from "@/components/export-artifacts";
import { MetricsStrip } from "@/components/metrics-strip";
import { Neighborhood3DCard } from "@/components/neighborhood-3d-card";
import { SimulationControls } from "@/components/simulation-controls";
import { TopologyMap } from "@/components/topology-map";
import { TrainingPanel } from "@/components/training-panel";
import { useSimulation } from "@/hooks/use-simulation";
import { useTraining } from "@/hooks/use-training";

export default function DashboardPage(): JSX.Element {
  const {
    simulationState,
    metrics,
    history,
    mode,
    setMode,
    autoRefresh,
    setAutoRefresh,
    isBusy,
    isProfilingCsv,
    isDerivingWeather,
    error,
    csvError,
    csvSchemas,
    csvProfile,
    derivedWeather,
    resetSession,
    stepSession,
    runSession,
    refreshSnapshot,
    analyzeCsv,
    deriveWeatherFromCsv,
  } = useSimulation();

  const {
    latestRun,
    latestComparison,
    rewardCurve,
    isTraining,
    error: trainingError,
    runTraining,
    runComparison,
    refreshArtifacts,
  } = useTraining();

  const [isRunningDemo, setIsRunningDemo] = useState<boolean>(false);

  const runFacultyDemoSequence = async () => {
    setIsRunningDemo(true);
    try {
      await resetSession({ seed: 2026 });
      await runSession(36);
      await runTraining({
        episodes: 24,
        steps_per_episode: 24,
        eval_episodes: 8,
        seed: 2026,
      });
      await runComparison({
        episodes: 8,
        steps_per_episode: 24,
        seed: 2026,
      });
      await Promise.all([refreshSnapshot(), refreshArtifacts()]);
    } finally {
      setIsRunningDemo(false);
    }
  };

  const topology = simulationState?.topology;
  const observation = simulationState?.observation;

  return (
    <main className="relative z-10 min-h-screen px-4 py-5 md:px-8 md:py-7">
      <header className="panel-surface mb-4 overflow-hidden px-6 py-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]">
              <Sparkles className="h-3.5 w-3.5" />
              Phase 3 Command Surface
            </p>
            <h1 className="mt-3 font-display text-3xl font-bold tracking-[0.04em] text-white md:text-4xl">
              Helios-Grid Mission Control Dashboard
            </h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-300 md:text-base">
              Show real simulation controls, live decentralized grid state, and PPO learning progress in one presentation-ready interface.
            </p>
          </div>

          <div className="grid gap-2 text-xs uppercase tracking-[0.15em] text-slate-300">
            <div className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-2">
              Episode: <span className="font-mono text-cyan-200">{simulationState?.episode_id ?? "-"}</span>
            </div>
            <div className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-2">
              Step: <span className="font-mono text-cyan-200">{simulationState?.step ?? "-"}</span>
            </div>
            <div className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-2">
              Weather Source:
              <span className="ml-2 font-mono text-cyan-200">
                {simulationState?.data_sources?.weather_data
                  ? simulationState.data_sources.weather_data.split(/[\\/]/).slice(-1)[0]
                  : "default"}
              </span>
            </div>
            <div className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-2">
              Mode:
              <span className="ml-2 inline-flex items-center gap-1 font-mono text-amber-200">
                {mode === "rule" ? <Cpu className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
                {mode === "rule" ? "Rule Live" : "PPO Preview"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {error ? (
        <div className="mb-4 rounded-md border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
          <p className="inline-flex items-center gap-2 font-semibold uppercase tracking-[0.12em]">
            <AlertTriangle className="h-4 w-4" />
            Simulation API Error
          </p>
          <p className="mt-1 text-rose-200">{error}</p>
        </div>
      ) : null}

      {trainingError ? (
        <div className="mb-4 rounded-md border border-rose-400/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
          <p className="inline-flex items-center gap-2 font-semibold uppercase tracking-[0.12em]">
            <AlertTriangle className="h-4 w-4" />
            Training API Error
          </p>
          <p className="mt-1 text-rose-200">{trainingError}</p>
        </div>
      ) : null}

      <MetricsStrip metrics={metrics} />

      <section className="mt-4 grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <div className="space-y-4">
          <SimulationControls
            mode={mode}
            setMode={setMode}
            autoRefresh={autoRefresh}
            setAutoRefresh={setAutoRefresh}
            isBusy={isBusy}
            isProfilingCsv={isProfilingCsv}
            isDerivingWeather={isDerivingWeather}
            isRunningDemo={isRunningDemo || isTraining}
            csvSchemas={csvSchemas}
            csvProfile={csvProfile}
            derivedWeather={derivedWeather}
            csvError={csvError}
            onReset={resetSession}
            onStep={stepSession}
            onRun={runSession}
            onRefresh={refreshSnapshot}
            onAnalyzeCsv={analyzeCsv}
            onDeriveWeatherFromCsv={deriveWeatherFromCsv}
            onRunDemoSequence={runFacultyDemoSequence}
          />
          <TopologyMap topology={topology} observation={observation} />
        </div>

        <div className="space-y-4">
          <Neighborhood3DCard topology={topology} />
          <TrainingPanel
            latestRun={latestRun}
            latestComparison={latestComparison}
            rewardCurve={rewardCurve}
            isTraining={isTraining}
            error={trainingError}
            onTrain={runTraining}
            onCompare={runComparison}
            onRefresh={refreshArtifacts}
          />
        </div>
      </section>

      <section className="mt-4">
        <EnergyCharts history={history} />
      </section>

      <ExportArtifacts
        metrics={metrics}
        history={history}
        latestRun={latestRun}
        latestComparison={latestComparison}
        rewardCurve={rewardCurve}
      />
    </main>
  );
}
