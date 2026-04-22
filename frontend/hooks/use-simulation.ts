"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, apiClient, DeriveWeatherInput, RunSimulationInput } from "@/lib/api-client";
import {
  CsvProfilePayload,
  CsvRole,
  CsvSchemasPayload,
  DerivedWeatherPayload,
  PolicyMode,
  SimulationMetrics,
  SimulationStateResponse,
  TrajectoryPoint,
} from "@/lib/types";

interface ResetSessionInput {
  seed?: number;
  weatherDataPath?: string;
}

interface UseSimulationState {
  simulationState: SimulationStateResponse | null;
  metrics: SimulationMetrics | null;
  history: TrajectoryPoint[];
  mode: PolicyMode;
  setMode: (mode: PolicyMode) => void;
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;
  isBusy: boolean;
  isProfilingCsv: boolean;
  isDerivingWeather: boolean;
  error: string | null;
  csvError: string | null;
  csvSchemas: CsvSchemasPayload | null;
  csvProfile: CsvProfilePayload | null;
  derivedWeather: DerivedWeatherPayload | null;
  resetSession: (input?: ResetSessionInput) => Promise<void>;
  stepSession: () => Promise<void>;
  runSession: (steps: number) => Promise<void>;
  refreshSnapshot: () => Promise<void>;
  analyzeCsv: (filePath: string, role: CsvRole) => Promise<void>;
  deriveWeatherFromCsv: (input: DeriveWeatherInput) => Promise<void>;
}

const HISTORY_LIMIT = 240;

export function useSimulation(): UseSimulationState {
  const [simulationState, setSimulationState] = useState<SimulationStateResponse | null>(null);
  const [metrics, setMetrics] = useState<SimulationMetrics | null>(null);
  const [history, setHistory] = useState<TrajectoryPoint[]>([]);
  const [mode, setMode] = useState<PolicyMode>("rule");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [isProfilingCsv, setIsProfilingCsv] = useState<boolean>(false);
  const [isDerivingWeather, setIsDerivingWeather] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvSchemas, setCsvSchemas] = useState<CsvSchemasPayload | null>(null);
  const [csvProfile, setCsvProfile] = useState<CsvProfilePayload | null>(null);
  const [derivedWeather, setDerivedWeather] = useState<DerivedWeatherPayload | null>(null);

  const captureError = useCallback((unknownError: unknown) => {
    if (unknownError instanceof ApiError) {
      setError(`${unknownError.message}`);
      return;
    }

    if (unknownError instanceof Error) {
      setError(unknownError.message);
      return;
    }

    setError("An unknown simulation error occurred.");
  }, []);

  const mergeTopology = useCallback(
    (incoming: SimulationStateResponse) => {
      setSimulationState((previous) => {
        if (incoming.topology) {
          return incoming;
        }

        if (previous?.topology) {
          return {
            ...incoming,
            topology: previous.topology,
          };
        }

        return incoming;
      });
    },
    [],
  );

  const captureCsvError = useCallback((unknownError: unknown) => {
    if (unknownError instanceof ApiError) {
      setCsvError(`${unknownError.message}`);
      return;
    }

    if (unknownError instanceof Error) {
      setCsvError(unknownError.message);
      return;
    }

    setCsvError("An unknown CSV profiling error occurred.");
  }, []);

  const refreshSnapshot = useCallback(async () => {
    try {
      const [statePayload, metricsPayload, historyPayload] = await Promise.all([
        apiClient.getSimulationState(false),
        apiClient.getSimulationMetrics(),
        apiClient.getSimulationHistory(HISTORY_LIMIT),
      ]);

      mergeTopology(statePayload);
      setMetrics(metricsPayload);
      setHistory(historyPayload);
      setError(null);
    } catch (unknownError) {
      captureError(unknownError);
    }
  }, [captureError, mergeTopology]);

  const resetSession = useCallback(
    async (input?: ResetSessionInput) => {
      setIsBusy(true);
      try {
        const payload = await apiClient.resetSimulation({
          seed: input?.seed,
          weather_data_path: input?.weatherDataPath,
        });
        mergeTopology(payload);
        setMetrics(payload.metrics);
        setHistory(payload.trajectory_point ? [payload.trajectory_point] : []);
        setError(null);
      } catch (unknownError) {
        captureError(unknownError);
      } finally {
        setIsBusy(false);
      }
    },
    [captureError, mergeTopology],
  );

  const analyzeCsv = useCallback(
    async (filePath: string, role: CsvRole) => {
      setIsProfilingCsv(true);
      try {
        const payload = await apiClient.profileCsvData({
          file_path: filePath,
          role,
          preview_rows: 5,
        });
        setCsvProfile(payload);
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsProfilingCsv(false);
      }
    },
    [captureCsvError],
  );

  const loadCsvSchemas = useCallback(async () => {
    try {
      const payload = await apiClient.getCsvSchemas();
      setCsvSchemas(payload);
      setCsvError(null);
    } catch (unknownError) {
      captureCsvError(unknownError);
    }
  }, [captureCsvError]);

  const deriveWeatherFromCsv = useCallback(
    async (input: DeriveWeatherInput) => {
      setIsDerivingWeather(true);
      try {
        const payload = await apiClient.deriveWeatherCsv(input);
        setDerivedWeather(payload);
        setCsvError(null);
      } catch (unknownError) {
        captureCsvError(unknownError);
      } finally {
        setIsDerivingWeather(false);
      }
    },
    [captureCsvError],
  );

  const stepSession = useCallback(async () => {
    setIsBusy(true);
    try {
      const payload = await apiClient.stepSimulation({
        use_autopilot: true,
        market_action: mode === "rule" ? 1 : 1,
      });
      mergeTopology(payload);
      setMetrics(payload.metrics);
      setHistory((prev) => [...prev, payload.trajectory_point].slice(-HISTORY_LIMIT));
      setError(null);
    } catch (unknownError) {
      captureError(unknownError);
    } finally {
      setIsBusy(false);
    }
  }, [captureError, mergeTopology, mode]);

  const runSession = useCallback(
    async (steps: number) => {
      const input: RunSimulationInput = {
        steps,
        use_autopilot: true,
        market_action: mode === "rule" ? 1 : 1,
      };

      setIsBusy(true);
      try {
        const payload = await apiClient.runSimulation(input);
        mergeTopology(payload.state);
        setMetrics(payload.metrics);
        setHistory((prev) => [...prev, ...payload.trajectory].slice(-HISTORY_LIMIT));
        setError(null);
      } catch (unknownError) {
        captureError(unknownError);
      } finally {
        setIsBusy(false);
      }
    },
    [captureError, mergeTopology, mode],
  );

  useEffect(() => {
    void resetSession();
  }, [resetSession]);

  useEffect(() => {
    void loadCsvSchemas();
  }, [loadCsvSchemas]);

  useEffect(() => {
    if (!autoRefresh) {
      return;
    }

    const interval = setInterval(() => {
      void refreshSnapshot();
    }, 2500);

    return () => {
      clearInterval(interval);
    };
  }, [autoRefresh, refreshSnapshot]);

  const stableMetrics = useMemo<SimulationMetrics | null>(() => {
    if (metrics) {
      return metrics;
    }

    return simulationState?.metrics ?? null;
  }, [metrics, simulationState]);

  return {
    simulationState,
    metrics: stableMetrics,
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
  };
}
