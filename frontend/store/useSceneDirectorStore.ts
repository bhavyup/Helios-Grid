"use client";

import { create } from "zustand";

export type DemoStatus = {
  running: boolean;
  phase: string;
  progress: number; // 0..1
  message?: string;
};

type SceneDirectorState = {
  // camera/selection
  selectedHouseId: number | null;

  // flow layers
  showGridFlows: boolean;
  showOrderFlows: boolean;
  showTradeFlows: boolean;
  maxFlows: number;

  // demo status
  demo: DemoStatus;

  setSelectedHouseId: (id: number | null) => void;

  setShowGridFlows: (v: boolean) => void;
  setShowOrderFlows: (v: boolean) => void;
  setShowTradeFlows: (v: boolean) => void;
  setMaxFlows: (v: number) => void;

  setDemo: (patch: Partial<DemoStatus>) => void;
  resetVisuals: () => void;
};

export const useSceneDirectorStore = create<SceneDirectorState>((set) => ({
  selectedHouseId: null,

  showGridFlows: true,
  showOrderFlows: false,
  showTradeFlows: true,
  maxFlows: 50,

  demo: { running: false, phase: "Idle", progress: 0 },

  setSelectedHouseId: (id) => set({ selectedHouseId: id }),

  setShowGridFlows: (v) => set({ showGridFlows: v }),
  setShowOrderFlows: (v) => set({ showOrderFlows: v }),
  setShowTradeFlows: (v) => set({ showTradeFlows: v }),
  setMaxFlows: (v) => set({ maxFlows: v }),

  setDemo: (patch) =>
    set((state) => ({
      demo: { ...state.demo, ...patch },
    })),

  resetVisuals: () =>
    set({
      selectedHouseId: null,
      showGridFlows: true,
      showOrderFlows: false,
      showTradeFlows: true,
      maxFlows: 50,
      demo: { running: false, phase: "Idle", progress: 0 },
    }),
}));