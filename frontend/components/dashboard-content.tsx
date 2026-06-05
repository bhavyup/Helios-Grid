"use client";

import { AlertTriangle, ArrowRight, Bot, Cpu, Sparkles } from "lucide-react";
import { useState } from "react";

import { EnergyCharts } from "@/components/energy-charts";
import { ExportArtifacts } from "@/components/export-artifacts";
import { MetricsStrip } from "@/components/metrics-strip";
import { Neighborhood3DCard } from "./neighborhood-3d-card";
import { SimulationControls } from "@/components/simulation-controls";
import { TopologyMap } from "@/components/topology-map";
import { TrainingPanel } from "@/components/training-panel";
import { useSimulation } from "@/hooks/useSimulation";
import { useTraining } from "@/hooks/use-training";
import { useWeather } from "@/hooks/useWeather";

export function DashboardContent(): JSX.Element {
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
    isUploadingWeather,
    error,
    csvError,
    csvSchemas,
    csvPaths,
    csvProfile,
    derivedWeather,
    uploadedWeather,
    resetSession,
    stepSession,
    runSession,
    refreshSnapshot,
    analyzeCsv,
    deriveWeatherFromCsv,
    uploadWeatherCsv,
    demoPhase,
    demoProgress,
    runDemoSequence,
    isRunningDemo,
  } = useSimulation();

  const {
    derivedHousehold,
    uploadedHousehold,
    isDerivingHousehold,
    isUploadingHousehold,
    deriveHouseholdFromCsv,
    uploadHouseholdCsv,
    derivedMarket,
    uploadedMarket,
    isDerivingMarket,
    isUploadingMarket,
    deriveMarketFromCsv,
    uploadMarketCsv,
  } = useWeather();

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

  function scrollToSection(sectionId: string) {
    const sectionElement = document.getElementById(sectionId);
    if (sectionElement) {
      sectionElement.scrollIntoView({ behavior: "smooth" });
    }
  }

  const topology = simulationState?.topology;
  const observation = simulationState?.observation;
  const latestInfo = simulationState?.latest_info;

  return (
    <main className="page-shell relative z-10 min-h-screen px-4 py-4 lg:px-6 lg:py-6">
      <div className="mx-auto flex w-full max-w-[1760px] flex-col gap-6">
        <header className="panel-surface relative overflo.w-hidden px-6 py-6 lg:px-8 lg:py-8">
          <div className="absolute inset-0 surface-grid" />
          <div className="absolute right-0 top-0 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgba(212,175,55,0.14),transparent_68%)] blur-3xl" />
          <div className="absolute left-12 top-10 h-40 w-40 rounded-full bg-[radial-gradient(circle,rgba(127,182,168,0.13),transparent_72%)] blur-3xl" />
          <div className="relative z-10 grid gap-8 xl:grid-cols-[1.12fr_0.88fr] xl:items-start">
            <div className="max-w-4xl h-full flex flex-col justify-between space-y-6">
              <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <p className="command-badge inline-flex items-center gap-2 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em]">
                  <Sparkles className="h-3.5 w-3.5" />
                  Weather refinery + live grid state + PPO analytics
                </p>
              </div>
              <h1 className="hero-display max-w-4xl text-5xl font-semibold tracking-[-0.045em] text-white md:text-7xl">
                Helios-Grid Mission Control
              </h1>
              <p className="section-copy max-w-3xl text-base md:text-lg">
                A calmer, more editorial control room for decentralized energy.
              </p>
              </div>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => scrollToSection("weather-refinery")}
                  className="inline-flex items-center gap-2 rounded-full border border-[rgba(212,175,55,0.34)] bg-[rgba(212,175,55,0.12)] px-5 py-3 text-sm font-semibold text-[#f6e7be] transition hover:bg-[rgba(212,175,55,0.18)]"
                >
                  Go to weather refinery
                  <ArrowRight className="h-4 w-4" />
                </button>
                <button
                  onClick={() => scrollToSection("learning-stack")}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
                >
                  Inspect learning stack
                </button>
                <button
                  onClick={() => scrollToSection("3d-scene")}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
                >
                  View 3D scene
                </button>
                <button
                  onClick={() => scrollToSection("exports")}
                  className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-5 py-3 text-sm font-semibold text-slate-100 transition hover:bg-white/[0.08]"
                >
                  Manage exports
                </button>
              </div>
            </div>

            <div className="grid gap-3 text-xs uppercase tracking-[0.18em] text-slate-300">
              {[
                ["Episode", simulationState?.episode_id ?? "-"],
                ["Step", simulationState?.step ?? "-"],
                [
                  "Weather source",
                  simulationState?.data_sources?.weather_data
                    ? simulationState.data_sources.weather_data
                        .split(/[\\/]/)
                        .slice(-1)[0]
                    : "default",
                ],
                ["Household source", simulationState?.data_sources?.household_data ? simulationState.data_sources.household_data.split(/[\\/]/).slice(-1)[0] : "default"],
                ["Market source", simulationState?.data_sources?.market_data ? simulationState.data_sources.market_data.split(/[\\/]/).slice(-1)[0] : "default"],
                ["Mode", mode === "rule" ? "Rule live" : "PPO preview"],
              ].map(([label, value], index) => (
                <div
                  key={label}
                  className="panel-frame rounded-[1.2rem] px-4 py-4 animate-reveal flex items-start"
                  style={{ animationDelay: `${index * 55}ms` }}
                >
                  <div className="whitespace-nowrap mr-2">{label}:</div>
                  <div className="flex items-start gap-1 font-mono text-[#f6e7be] break-all">
                    {label === "Mode" ? (
                      mode === "rule" ? (
                        <Cpu className="h-3.5 w-3.5 shrink-0 mt-[0.1rem]" />
                      ) : (
                        <Bot className="h-3.5 w-3.5 shrink-0 mt-[0.1rem]" />
                      )
                    ) : null}
                    {value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </header>

        {error ? (
          <div className="rounded-[1.2rem] border border-rose-300/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            <p className="inline-flex items-center gap-2 font-semibold uppercase tracking-[0.12em]">
              <AlertTriangle className="h-4 w-4" />
              Simulation API error
            </p>
            <p className="mt-1 text-rose-200">{error}</p>
          </div>
        ) : null}

        {trainingError ? (
          <div className="rounded-[1.2rem] border border-rose-300/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
            <p className="inline-flex items-center gap-2 font-semibold uppercase tracking-[0.12em]">
              <AlertTriangle className="h-4 w-4" />
              Training API error
            </p>
            <p className="mt-1 text-rose-200">{trainingError}</p>
          </div>
        ) : null}

        <MetricsStrip metrics={metrics} />

        <section className="" id="weather-refinery">
          <div className="space-y-6">
            <SimulationControls
              mode={mode}
              setMode={setMode}
              autoRefresh={autoRefresh}
              setAutoRefresh={setAutoRefresh}
              isBusy={isBusy}
              isProfilingCsv={isProfilingCsv}
              isDerivingWeather={isDerivingWeather}
              isUploadingWeather={isUploadingWeather}
              isRunningDemo={isRunningDemo || isTraining}
              csvSchemas={csvSchemas}
              csvProfile={csvProfile}
              derivedWeather={derivedWeather}
              uploadedWeather={uploadedWeather}
              csvError={csvError}
              csvPaths={csvPaths}
              onReset={resetSession}
              onStep={stepSession}
              onRun={runSession}
              onRefresh={refreshSnapshot}
              onAnalyzeCsv={analyzeCsv}
              onDeriveWeatherFromCsv={deriveWeatherFromCsv}
              onUploadWeatherCsv={uploadWeatherCsv}
              demoPhase={demoPhase}
              demoProgress={demoProgress}
              onRunDemoSequence={runDemoSequence}
              derivedHousehold={derivedHousehold}
              uploadedHousehold={uploadedHousehold}
              onDeriveHouseholdFromCsv={deriveHouseholdFromCsv}
              onUploadHouseholdCsv={uploadHouseholdCsv}
              derivedMarket={derivedMarket}
              uploadedMarket={uploadedMarket}
              onDeriveMarketFromCsv={deriveMarketFromCsv}
              onUploadMarketCsv={uploadMarketCsv}
              isDerivingHousehold={isDerivingHousehold}
              isUploadingHousehold={isUploadingHousehold}
              isDerivingMarket={isDerivingMarket}
              isUploadingMarket={isUploadingMarket}
            />

            <EnergyCharts history={history} />
          </div>

          <div className="space-y-12 mt-6" id="learning-stack">
            <div className="grid gap-6 grid-cols-2 h-full">
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
              <TopologyMap topology={topology} observation={observation} />
            </div>
          </div>
          <div id="3d-scene" className="mt-6">
            <Neighborhood3DCard
              topology={topology}
              observation={observation}
              latestInfo={latestInfo}
              step={simulationState?.step}
            />
          </div>
        </section>

        <div id="exports" className="space-y-4">
          <p className="section-eyebrow px-1">Export deck</p>
          <ExportArtifacts
            metrics={metrics}
            history={history}
            latestRun={latestRun}
            latestComparison={latestComparison}
            rewardCurve={rewardCurve}
          />
        </div>
      </div>
    </main>
  );
}
