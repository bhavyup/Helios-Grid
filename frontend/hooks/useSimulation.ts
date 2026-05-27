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

import { useMetrics } from "@/hooks/useMetrics";
import { useSimulationPolling } from "@/hooks/useSimulationPolling";
import { useWeather } from "@/hooks/useWeather";
import { useSimulationStore } from "@/store/useSimulationStore";

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
  const simulationState = useSimulationStore((state) => state.simulationState);
  const setSimulationState = useSimulationStore(
    (state) => state.setSimulationState,
  );
  const mode = useSimulationStore((state) => state.mode);
  const setMode = useSimulationStore((state) => state.setMode);
  const autoRefreshState = useSimulationStore((state) => state.autoRefresh);
  const setAutoRefreshState = useSimulationStore(
    (state) => state.setAutoRefresh,
  );
  const [isBusy, setIsBusy] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const {
    metrics,
    history,
    updateFromSnapshot,
    updateFromStep,
    updateFromRun,
    updateFromPolling,
  } = useMetrics({ historyLimit: HISTORY_LIMIT });

  const {
    isProfilingCsv,
    isDerivingWeather,
    csvError,
    csvSchemas,
    csvProfile,
    derivedWeather,
    analyzeCsv,
    deriveWeatherFromCsv,
  } = useWeather({ autoLoadSchemas: true });

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

  const mergeTopology = useCallback((incoming: SimulationStateResponse) => {
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
  }, []);

  const refreshSnapshot = useCallback(async () => {
    try {
      const [statePayload, metricsPayload, historyPayload] = await Promise.all([
        apiClient.getSimulationState(false),
        apiClient.getSimulationMetrics(),
        apiClient.getSimulationHistory(HISTORY_LIMIT),
      ]);

      mergeTopology(statePayload);
      updateFromPolling(metricsPayload, historyPayload);
      setError(null);
    } catch (unknownError) {
      captureError(unknownError);
    }
  }, [captureError, mergeTopology, updateFromPolling]);

  const { autoRefresh, setAutoRefresh } = useSimulationPolling({
    onRefresh: refreshSnapshot,
    intervalMs: 2500,
    autoRefresh: autoRefreshState,
    setAutoRefresh: setAutoRefreshState,
  });

  const resetSession = useCallback(
    async (input?: ResetSessionInput) => {
      setIsBusy(true);
      try {
        const payload = await apiClient.resetSimulation({
          seed: input?.seed,
          weather_data_path: input?.weatherDataPath,
        });
        mergeTopology(payload);
        updateFromSnapshot(payload);
        setError(null);
      } catch (unknownError) {
        captureError(unknownError);
      } finally {
        setIsBusy(false);
      }
    },
    [captureError, mergeTopology, updateFromSnapshot],
  );

  const stepSession = useCallback(async () => {
    setIsBusy(true);
    try {
      const payload = await apiClient.stepSimulation({
        use_autopilot: true,
        market_action: mode === "rule" ? 1 : 1,
      });
      mergeTopology(payload);
      updateFromStep(payload);
      setError(null);
    } catch (unknownError) {
      captureError(unknownError);
    } finally {
      setIsBusy(false);
    }
  }, [captureError, mergeTopology, mode, updateFromStep]);

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
        updateFromRun(payload);
        setError(null);
      } catch (unknownError) {
        captureError(unknownError);
      } finally {
        setIsBusy(false);
      }
    },
    [captureError, mergeTopology, mode, updateFromRun],
  );

  useEffect(() => {
    void resetSession();
  }, [resetSession]);

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
