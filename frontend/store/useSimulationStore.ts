"use client";

import { create } from "zustand";

import {
  PolicyMode,
  SimulationMetrics,
  SimulationStateResponse,
  TrajectoryPoint,
} from "@/lib/types";

interface SimulationStoreState {
  simulationState: SimulationStateResponse | null;
  metrics: SimulationMetrics | null;
  history: TrajectoryPoint[];
  mode: PolicyMode;
  autoRefresh: boolean;
  setSimulationState: (state: SimulationStateResponse | null) => void;
  setMetrics: (metrics: SimulationMetrics | null) => void;
  setHistory: (history: TrajectoryPoint[]) => void;
  setMode: (mode: PolicyMode) => void;
  setAutoRefresh: (value: boolean) => void;
  resetSimulation: () => void;
}

export const useSimulationStore = create<SimulationStoreState>((set) => ({
  simulationState: null,
  metrics: null,
  history: [],
  mode: "rule",
  autoRefresh: false,
  setSimulationState: (state) => set({ simulationState: state }),
  setMetrics: (metrics) => set({ metrics }),
  setHistory: (history) => set({ history }),
  setMode: (mode) => set({ mode }),
  setAutoRefresh: (autoRefresh) => set({ autoRefresh }),
  resetSimulation: () =>
    set({
      simulationState: null,
      metrics: null,
      history: [],
    }),
}));
