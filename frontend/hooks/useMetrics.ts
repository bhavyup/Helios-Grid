"use client";

import { useCallback } from "react";

import {
  SimulationMetrics,
  SimulationRunResponse,
  SimulationStateResponse,
  TrajectoryPoint,
} from "@/lib/types";
import { useSimulationStore } from "@/store/useSimulationStore";

interface UseMetricsOptions {
  historyLimit?: number;
}

interface UseMetricsState {
  metrics: SimulationMetrics | null;
  history: TrajectoryPoint[];
  setMetrics: (metrics: SimulationMetrics | null) => void;
  setHistory: (history: TrajectoryPoint[]) => void;
  updateFromSnapshot: (payload: SimulationStateResponse) => void;
  updateFromStep: (payload: SimulationStateResponse) => void;
  updateFromRun: (payload: SimulationRunResponse) => void;
  updateFromPolling: (
    metricsPayload: SimulationMetrics,
    historyPayload: TrajectoryPoint[],
  ) => void;
}

const DEFAULT_HISTORY_LIMIT = 240;

export function useMetrics(options: UseMetricsOptions = {}): UseMetricsState {
  const historyLimit = options.historyLimit ?? DEFAULT_HISTORY_LIMIT;
  const metrics = useSimulationStore((state) => state.metrics);
  const history = useSimulationStore((state) => state.history);
  const setMetrics = useSimulationStore((state) => state.setMetrics);
  const setHistory = useSimulationStore((state) => state.setHistory);

  const updateFromSnapshot = useCallback(
    (payload: SimulationStateResponse) => {
      setMetrics(payload.metrics);
      setHistory(payload.trajectory_point ? [payload.trajectory_point] : []);
    },
    [setHistory, setMetrics],
  );

  const updateFromStep = useCallback(
    (payload: SimulationStateResponse) => {
      setMetrics(payload.metrics);
      if (payload.trajectory_point) {
        setHistory((prev) =>
          [...prev, payload.trajectory_point].slice(-historyLimit),
        );
      }
    },
    [historyLimit, setHistory, setMetrics],
  );

  const updateFromRun = useCallback(
    (payload: SimulationRunResponse) => {
      setMetrics(payload.metrics);
      setHistory((prev) =>
        [...prev, ...payload.trajectory].slice(-historyLimit),
      );
    },
    [historyLimit, setHistory, setMetrics],
  );

  const updateFromPolling = useCallback(
    (metricsPayload: SimulationMetrics, historyPayload: TrajectoryPoint[]) => {
      setMetrics(metricsPayload);
      setHistory(historyPayload);
    },
    [setHistory, setMetrics],
  );

  return {
    metrics,
    history,
    setMetrics,
    setHistory,
    updateFromSnapshot,
    updateFromStep,
    updateFromRun,
    updateFromPolling,
  };
}
