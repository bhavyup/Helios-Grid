"use client";

import { Database, Loader2, Play, PlayCircle, RefreshCcw, RotateCcw, StepForward } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { CsvProfilePayload, CsvRole, CsvSchemasPayload, DerivedWeatherPayload, PolicyMode } from "@/lib/types";

interface SimulationControlsProps {
  mode: PolicyMode;
  setMode: (mode: PolicyMode) => void;
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;
  isBusy: boolean;
  isProfilingCsv: boolean;
  isDerivingWeather: boolean;
  isRunningDemo: boolean;
  csvSchemas: CsvSchemasPayload | null;
  csvProfile: CsvProfilePayload | null;
  derivedWeather: DerivedWeatherPayload | null;
  csvError: string | null;
  onReset: (input?: { seed?: number; weatherDataPath?: string }) => Promise<void>;
  onStep: () => Promise<void>;
  onRun: (steps: number) => Promise<void>;
  onRefresh: () => Promise<void>;
  onAnalyzeCsv: (filePath: string, role: CsvRole) => Promise<void>;
  onDeriveWeatherFromCsv: (input: {
    file_path: string;
    solar_column: string;
    wind_column: string;
    timestamp_column?: string;
    normalize_signals?: boolean;
  }) => Promise<void>;
  onRunDemoSequence: () => Promise<void>;
}

export function SimulationControls({
  mode,
  setMode,
  autoRefresh,
  setAutoRefresh,
  isBusy,
  isProfilingCsv,
  isDerivingWeather,
  isRunningDemo,
  csvSchemas,
  csvProfile,
  derivedWeather,
  csvError,
  onReset,
  onStep,
  onRun,
  onRefresh,
  onAnalyzeCsv,
  onDeriveWeatherFromCsv,
  onRunDemoSequence,
}: SimulationControlsProps): JSX.Element {
  const [runSteps, setRunSteps] = useState<number>(12);
  const [seedInput, setSeedInput] = useState<string>("");
  const [csvPath, setCsvPath] = useState<string>("");
  const [csvRole, setCsvRole] = useState<CsvRole>("auto");
  const [solarColumnInput, setSolarColumnInput] = useState<string>("");
  const [windColumnInput, setWindColumnInput] = useState<string>("");
  const [timestampColumnInput, setTimestampColumnInput] = useState<string>("");

  const suggestedColumns = useMemo(() => {
    const columns = csvProfile?.columns ?? [];
    const solar = columns.find((columnName) => /solar(_|.*)generation_actual|solar_profile|solar/i.test(columnName));
    const wind = columns.find((columnName) => /wind(_|.*)generation_actual|wind_profile|wind/i.test(columnName));
    const timestamp = columns.find((columnName) => /utc_timestamp|timestamp/i.test(columnName));
    return { solar, wind, timestamp };
  }, [csvProfile]);

  useEffect(() => {
    if (!csvProfile) {
      return;
    }

    if (!solarColumnInput && suggestedColumns.solar) {
      setSolarColumnInput(suggestedColumns.solar);
    }
    if (!windColumnInput && suggestedColumns.wind) {
      setWindColumnInput(suggestedColumns.wind);
    }
    if (!timestampColumnInput && suggestedColumns.timestamp) {
      setTimestampColumnInput(suggestedColumns.timestamp);
    }
  }, [csvProfile, suggestedColumns, solarColumnInput, windColumnInput, timestampColumnInput]);

  const handleReset = async () => {
    const maybeSeed = seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN) ? maybeSeed : undefined;
    const parsedWeatherPath = csvPath.trim().length > 0 ? csvPath.trim() : undefined;
    await onReset({ seed: parsedSeed, weatherDataPath: parsedWeatherPath });
  };

  const handleAnalyzeCsv = async () => {
    const resolvedPath = csvPath.trim();
    if (!resolvedPath) {
      return;
    }
    await onAnalyzeCsv(resolvedPath, csvRole);
  };

  const handleResetWithProfile = async () => {
    if (!csvProfile?.can_use_now) {
      return;
    }

    const maybeSeed = seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN) ? maybeSeed : undefined;
    await onReset({ seed: parsedSeed, weatherDataPath: csvProfile.resolved_path });
  };

  const handleDeriveWeather = async () => {
    const filePath = csvProfile?.resolved_path ?? csvPath.trim();
    const solarColumn = solarColumnInput.trim();
    const windColumn = windColumnInput.trim();
    const timestampColumn = timestampColumnInput.trim();

    if (!filePath || !solarColumn || !windColumn) {
      return;
    }

    await onDeriveWeatherFromCsv({
      file_path: filePath,
      solar_column: solarColumn,
      wind_column: windColumn,
      timestamp_column: timestampColumn.length > 0 ? timestampColumn : undefined,
      normalize_signals: true,
    });
  };

  const handleResetWithDerivedWeather = async () => {
    if (!derivedWeather?.output_file_path) {
      return;
    }

    const maybeSeed = seedInput.trim().length > 0 ? Number(seedInput) : undefined;
    const parsedSeed = Number.isFinite(maybeSeed ?? NaN) ? maybeSeed : undefined;
    await onReset({ seed: parsedSeed, weatherDataPath: derivedWeather.output_file_path });
  };

  return (
    <section className="panel-surface px-5 py-5">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="panel-title">Simulation Controls</h2>
          <p className="mt-1 text-xs text-slate-400">Run, step, reset and monitor decentralized grid behavior.</p>
        </div>
        <button
          type="button"
          onClick={() => void onRefresh()}
          className="inline-flex items-center gap-2 rounded-md border border-slate-700/80 bg-slate-800/70 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-cyan-400/70 hover:text-cyan-200 active:scale-[0.98]"
        >
          <RefreshCcw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      <div className="grid gap-4 lg:grid-cols-4">
        <label className="space-y-2 text-xs text-slate-300">
          <span className="uppercase tracking-[0.15em] text-slate-400">Policy Mode</span>
          <select
            value={mode}
            onChange={(event) => setMode(event.target.value as PolicyMode)}
            className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
          >
            <option value="rule">Rule Mode (Live)</option>
            <option value="ppo-preview">PPO Preview (Analytics)</option>
          </select>
        </label>

        <label className="space-y-2 text-xs text-slate-300">
          <span className="uppercase tracking-[0.15em] text-slate-400">Run Steps</span>
          <input
            type="number"
            min={1}
            max={240}
            value={runSteps}
            onChange={(event) => setRunSteps(Math.max(1, Number(event.target.value) || 1))}
            className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
          />
        </label>

        <label className="space-y-2 text-xs text-slate-300">
          <span className="uppercase tracking-[0.15em] text-slate-400">Reset Seed (optional)</span>
          <input
            type="number"
            placeholder="42"
            value={seedInput}
            onChange={(event) => setSeedInput(event.target.value)}
            className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
          />
        </label>

        <label className="flex items-end gap-2 rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-2">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(event) => setAutoRefresh(event.target.checked)}
            className="h-4 w-4 rounded border-slate-600 bg-slate-800 text-cyan-500"
          />
          <span className="text-xs uppercase tracking-[0.15em] text-slate-300">Auto Poll</span>
        </label>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <button
          type="button"
          disabled={isBusy}
          onClick={() => void onStep()}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-cyan-400/70 bg-cyan-500/15 px-3 py-2 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <StepForward className="h-4 w-4" />
          Step
        </button>

        <button
          type="button"
          disabled={isBusy}
          onClick={() => void onRun(runSteps)}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-emerald-400/70 bg-emerald-500/15 px-3 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Play className="h-4 w-4" />
          Run {runSteps} Steps
        </button>

        <button
          type="button"
          disabled={isBusy}
          onClick={() => void handleReset()}
          className="inline-flex items-center justify-center gap-2 rounded-md border border-amber-300/70 bg-amber-400/15 px-3 py-2 text-sm font-semibold text-amber-100 transition hover:bg-amber-400/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <RotateCcw className="h-4 w-4" />
          Reset Episode
        </button>
      </div>

      <div className="mt-3">
        <button
          type="button"
          disabled={isBusy || isRunningDemo}
          onClick={() => void onRunDemoSequence()}
          className="inline-flex w-full items-center justify-center gap-2 rounded-md border border-fuchsia-300/70 bg-fuchsia-500/15 px-3 py-2 text-sm font-semibold text-fuchsia-100 transition hover:bg-fuchsia-500/25 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isRunningDemo ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlayCircle className="h-4 w-4" />}
          {isRunningDemo ? "Running Demo Sequence..." : "Run Faculty Demo Sequence"}
        </button>
      </div>

      <div className="mt-6 rounded-xl border border-slate-700/80 bg-slate-950/60 px-4 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="inline-flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.14em] text-cyan-200">
              <Database className="h-4 w-4" />
              CSV Intake Analyzer
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Profile your CSV and verify if it can power weather, household, or market data pathways.
            </p>
          </div>
          <button
            type="button"
            disabled={isProfilingCsv || csvPath.trim().length === 0}
            onClick={() => void handleAnalyzeCsv()}
            className="inline-flex items-center gap-2 rounded-md border border-cyan-400/60 bg-cyan-500/15 px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-100 transition hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isProfilingCsv ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCcw className="h-3.5 w-3.5" />}
            Analyze CSV
          </button>
        </div>

        <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_220px]">
          <label className="space-y-1 text-xs text-slate-300">
            <span className="uppercase tracking-[0.15em] text-slate-400">CSV Path</span>
            <input
              type="text"
              placeholder="backend/data/weather_data/my_weather.csv"
              value={csvPath}
              onChange={(event) => setCsvPath(event.target.value)}
              className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
            />
          </label>

          <label className="space-y-1 text-xs text-slate-300">
            <span className="uppercase tracking-[0.15em] text-slate-400">Role Check</span>
            <select
              value={csvRole}
              onChange={(event) => setCsvRole(event.target.value as CsvRole)}
              className="w-full rounded-md border border-slate-700 bg-slate-900/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/70"
            >
              <option value="auto">Auto Detect</option>
              <option value="weather">Weather</option>
              <option value="household">Household</option>
              <option value="market">Market</option>
            </select>
          </label>
        </div>

        {csvError ? (
          <p className="mt-3 rounded-md border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">{csvError}</p>
        ) : null}

        {csvProfile ? (
          <div className="mt-3 space-y-3 rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-3">
            <div className="grid gap-2 text-xs text-slate-300 sm:grid-cols-3">
              <p>
                <span className="text-slate-400">Rows:</span> {csvProfile.rows}
              </p>
              <p>
                <span className="text-slate-400">Columns:</span> {csvProfile.column_count}
              </p>
              <p>
                <span className="text-slate-400">Detected Role:</span> {csvProfile.inferred_role}
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              {csvProfile.columns.slice(0, 8).map((columnName) => (
                <span
                  key={columnName}
                  className="rounded-full border border-slate-600/70 bg-slate-950/70 px-2 py-1 text-[11px] text-slate-200"
                >
                  {columnName}
                </span>
              ))}
            </div>

            <p className="text-xs text-cyan-100">{csvProfile.usage_recommendation}</p>

            {csvProfile.can_use_now ? (
              <button
                type="button"
                disabled={isBusy}
                onClick={() => void handleResetWithProfile()}
                className="inline-flex items-center gap-2 rounded-md border border-emerald-400/60 bg-emerald-500/15 px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-emerald-100 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Play className="h-3.5 w-3.5" />
                Use As Weather Dataset + Reset
              </button>
            ) : null}

            {!csvProfile.can_use_now ? (
              <div className="space-y-2 rounded-md border border-cyan-500/30 bg-cyan-500/5 p-3">
                <p className="text-[11px] uppercase tracking-[0.13em] text-cyan-200">
                  Derive Weather CSV From Wide Dataset
                </p>
                <div className="grid gap-2 lg:grid-cols-3">
                  <input
                    type="text"
                    value={solarColumnInput}
                    onChange={(event) => setSolarColumnInput(event.target.value)}
                    placeholder="solar column"
                    className="w-full rounded-md border border-slate-700 bg-slate-950/80 px-2 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/70"
                  />
                  <input
                    type="text"
                    value={windColumnInput}
                    onChange={(event) => setWindColumnInput(event.target.value)}
                    placeholder="wind column"
                    className="w-full rounded-md border border-slate-700 bg-slate-950/80 px-2 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/70"
                  />
                  <input
                    type="text"
                    value={timestampColumnInput}
                    onChange={(event) => setTimestampColumnInput(event.target.value)}
                    placeholder="timestamp column (optional)"
                    className="w-full rounded-md border border-slate-700 bg-slate-950/80 px-2 py-2 text-xs text-slate-100 outline-none transition focus:border-cyan-400/70"
                  />
                </div>

                <button
                  type="button"
                  disabled={isDerivingWeather || !solarColumnInput.trim() || !windColumnInput.trim()}
                  onClick={() => void handleDeriveWeather()}
                  className="inline-flex items-center gap-2 rounded-md border border-cyan-400/60 bg-cyan-500/15 px-3 py-2 text-xs font-semibold uppercase tracking-[0.1em] text-cyan-100 transition hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isDerivingWeather ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Database className="h-3.5 w-3.5" />}
                  {isDerivingWeather ? "Deriving..." : "Create Weather CSV"}
                </button>

                {derivedWeather ? (
                  <div className="rounded-md border border-emerald-400/30 bg-emerald-500/5 px-2 py-2 text-xs text-emerald-100">
                    <p>Derived file: {derivedWeather.output_file_path}</p>
                    <p className="mt-1">Rows: {derivedWeather.rows}</p>
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void handleResetWithDerivedWeather()}
                      className="mt-2 inline-flex items-center gap-2 rounded-md border border-emerald-400/60 bg-emerald-500/15 px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-emerald-100 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <Play className="h-3.5 w-3.5" />
                      Use Derived Weather + Reset
                    </button>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}

        {csvSchemas ? (
          <p className="mt-3 text-[11px] text-slate-500">
            Runtime-ready now: {csvSchemas.weather.runtime_supported_now ? "weather" : "none"}. Planned next: household and market wiring.
          </p>
        ) : null}
      </div>

      <p className="mt-4 text-xs text-slate-400">
        PPO Preview mode is analytics-first in this phase. Live simulation stepping remains on rule-autopilot while PPO progress is shown in the training panel.
      </p>
    </section>
  );
}
