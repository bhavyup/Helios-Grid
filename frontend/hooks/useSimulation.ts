"use client";

import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { useSceneDirectorStore } from "@/store/useSceneDirectorStore";

import {
  ApiError,
  apiClient,
  DeriveWeatherInput,
  RunSimulationInput,
} from "@/lib/api-client";
import {
  CsvProfilePayload,
  CsvRole,
  CsvSchemasPayload,
  DerivedWeatherPayload,
  PolicyMode,
  SimulationMetrics,
  SimulationStateResponse,
  TrajectoryPoint,
  UploadedWeatherPayload,
} from "@/lib/types";

import { useMetrics } from "@/hooks/useMetrics";
import { useSimulationPolling } from "@/hooks/useSimulationPolling";
import { useWeather } from "@/hooks/useWeather";
import { useSimulationStore } from "@/store/useSimulationStore";

interface ResetSessionInput {
  seed?: number;
  weatherDataPath?: string;
  householdDataPath?: string;
  marketDataPath?: string;
  numHouseholds?: number;
  maxEpisodeSteps?: number;
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
  isUploadingWeather: boolean;
  error: string | null;
  csvError: string | null;
  csvSchemas: CsvSchemasPayload | null;
  csvPaths: {
    paths: { path: string; kind: string; label: string }[];
    count: number;
  } | null;
  csvProfile: CsvProfilePayload | null;
  derivedWeather: DerivedWeatherPayload | null;
  uploadedWeather: UploadedWeatherPayload | null;
  resetSession: (input?: ResetSessionInput) => Promise<void>;
  stepSession: () => Promise<void>;
  runSession: (steps: number) => Promise<void>;
  refreshSnapshot: () => Promise<void>;
  analyzeCsv: (filePath: string, role: CsvRole) => Promise<void>;
  deriveWeatherFromCsv: (input: DeriveWeatherInput) => Promise<void>;
  uploadWeatherCsv: (input: { file: File }) => Promise<void>;
  isRunningDemo: boolean;
  demoPhase: string;
  demoProgress: number;
  runDemoSequence: () => Promise<void>;
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

  const [isRunningDemo, setIsRunningDemo] = useState(false);
  const [demoPhase, setDemoPhase] = useState("Idle");
  const [demoProgress, setDemoProgress] = useState(0);

  const demoCancelRef = useRef(false);

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
    isUploadingWeather,
    csvError,
    csvSchemas,
    csvPaths,
    csvProfile,
    derivedWeather,
    uploadedWeather,
    analyzeCsv,
    deriveWeatherFromCsv,
    uploadWeatherCsv,
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
    const previous = useSimulationStore.getState().simulationState;
    if (incoming.topology) {
      setSimulationState(incoming);
      return;
    }

    if (previous?.topology) {
      setSimulationState({
        ...incoming,
        topology: previous.topology,
      });
      return;
    }

    setSimulationState(incoming);
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
          num_households: input?.numHouseholds,
          max_episode_steps: input?.maxEpisodeSteps,
          weather_data_path: input?.weatherDataPath,
          household_data_path: input?.householdDataPath,
          market_data_path: input?.marketDataPath,
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

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const pickTopHouse = (
    houseStates: number[][],
    index: number,
  ): number | null => {
    // house id in topology is 1..N, lot_index corresponds to row in house_states
    if (!houseStates?.length) return null;
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < houseStates.length; i += 1) {
      const v = Number(houseStates[i]?.[index] ?? 0);
      if (v > bestVal) {
        bestVal = v;
        best = i;
      }
    }
    return best + 1;
  };

  const pickTopTrader = (marketSnapshot: any): number | null => {
    const trades = marketSnapshot?.p2p_trades;
    if (!Array.isArray(trades) || trades.length === 0) return null;

    const volByHouse = new Map<number, number>();
    for (const t of trades) {
      const buyer = Number(t?.buyer_household_id);
      const seller = Number(t?.seller_household_id);
      const qty = Number(t?.quantity ?? 0);
      if (Number.isFinite(buyer))
        volByHouse.set(buyer, (volByHouse.get(buyer) ?? 0) + qty);
      if (Number.isFinite(seller))
        volByHouse.set(seller, (volByHouse.get(seller) ?? 0) + qty);
    }

    let bestId: number | null = null;
    let bestVol = -Infinity;
    for (const [id, vol] of volByHouse.entries()) {
      if (vol > bestVol) {
        bestVol = vol;
        bestId = id;
      }
    }
    return bestId;
  };

  const runDemoSequence = useCallback(async () => {
    // toggle behavior: if running, request cancel
    if (isRunningDemo) {
      demoCancelRef.current = true;
      useSceneDirectorStore.getState().setDemo({ message: "Stopping..." });
      return;
    }

    demoCancelRef.current = false;
    setIsRunningDemo(true);

    const director = useSceneDirectorStore.getState();
    director.setDemo({
      running: true,
      phase: "Initializing",
      progress: 0,
      message: "",
    });

    const previousAutoRefresh = useSimulationStore.getState().autoRefresh;
    setAutoRefresh(false); // stop polling during the scripted run

    try {
      // Reset into a stable demo setup
      setDemoPhase("Reset");
      setDemoProgress(0);
      director.resetVisuals();
      director.setDemo({ phase: "Reset", progress: 0 });

      const demoWeatherPath =
        derivedWeather?.output_file_path ??
        uploadedWeather?.resolved_path ??
        undefined;

      await resetSession({
        seed: 42,
        numHouseholds: 64,
        maxEpisodeSteps: 4000,
        weatherDataPath: demoWeatherPath,
      });

      // Small pause so the audience sees the reset “snap”
      await sleep(900);

      // Helper: one step (slow) so 3D has time to animate
      const stepOnce = async (paceMs: number) => {
        const payload = await apiClient.stepSimulation({
          use_autopilot: true,
          market_action: 1,
        });
        mergeTopology(payload);
        updateFromStep(payload);
        setError(null);
        await sleep(paceMs);
        return payload;
      };

      // PHASE 1 — Morning / demand & baseline
      setDemoPhase("Baseline demand");
      director.setDemo({
        phase: "Baseline demand",
        progress: 0,
        message: "Highlighting top consumers",
      });
      director.setShowGridFlows(true);
      director.setShowOrderFlows(false);
      director.setShowTradeFlows(false);
      director.setMaxFlows(35);

      for (let i = 0; i < 60; i += 1) {
        if (demoCancelRef.current) break;
        const payload = await stepOnce(380);

        const houseStates = payload?.observation?.house_states as
          | number[][]
          | undefined;
        const topConsumerId = houseStates ? pickTopHouse(houseStates, 1) : null; // consumption
        if (topConsumerId && i % 8 === 0)
          director.setSelectedHouseId(topConsumerId);

        const prog = i / 60;
        setDemoProgress(prog);
        director.setDemo({ progress: prog });
      }

      if (!demoCancelRef.current) await sleep(600);

      // PHASE 2 — Order depth (briefly show orders)
      setDemoPhase("Market depth (orders)");
      director.setDemo({
        phase: "Market depth (orders)",
        progress: 0,
        message: "Orders layer ON briefly",
      });
      director.setShowGridFlows(false);
      director.setShowOrderFlows(true);
      director.setShowTradeFlows(false);
      director.setMaxFlows(60);

      for (let i = 0; i < 30; i += 1) {
        if (demoCancelRef.current) break;
        const payload = await stepOnce(300);

        const houseStates = payload?.observation?.house_states as
          | number[][]
          | undefined;
        if (houseStates && i % 6 === 0) {
          const topBuyer = pickTopHouse(houseStates, 6);
          const topSeller = pickTopHouse(houseStates, 7);
          director.setSelectedHouseId(i % 12 === 0 ? topSeller : topBuyer);
        }

        const prog = i / 30;
        director.setDemo({ progress: prog });
      }

      if (!demoCancelRef.current) await sleep(700);

      // PHASE 3 — P2P trades peak
      setDemoPhase("P2P trading (matched trades)");
      director.setDemo({
        phase: "P2P trading (matched trades)",
        progress: 0,
        message: "Trades layer ON + max flows boost",
      });
      director.setShowGridFlows(false);
      director.setShowOrderFlows(false);
      director.setShowTradeFlows(true);
      director.setMaxFlows(95);

      for (let i = 0; i < 90; i += 1) {
        if (demoCancelRef.current) break;
        const payload = await stepOnce(260);

        const ms = payload?.latest_info?.market_snapshot;
        const topTrader = pickTopTrader(ms);
        if (topTrader && i % 7 === 0) director.setSelectedHouseId(topTrader);

        // after “wow” moment, reduce clutter
        if (i === 18) director.setMaxFlows(65);

        const prog = i / 90;
        director.setDemo({ progress: prog });
      }

      if (!demoCancelRef.current) await sleep(700);

      // PHASE 4 — Evening / grid reliance
      setDemoPhase("Evening: grid reliance");
      director.setDemo({
        phase: "Evening: grid reliance",
        progress: 0,
        message: "Grid layer ON, highlight importers",
      });
      director.setShowGridFlows(true);
      director.setShowOrderFlows(false);
      director.setShowTradeFlows(true);
      director.setMaxFlows(55);

      for (let i = 0; i < 60; i += 1) {
        if (demoCancelRef.current) break;
        const payload = await stepOnce(420);

        const houseStates = payload?.observation?.house_states as
          | number[][]
          | undefined;
        const topImporter = houseStates ? pickTopHouse(houseStates, 5) : null; // grid import
        if (topImporter && i % 9 === 0)
          director.setSelectedHouseId(topImporter);

        const prog = i / 60;
        director.setDemo({ progress: prog });
      }

      // OUTRO — release control
      director.setDemo({
        phase: demoCancelRef.current ? "Stopped" : "Complete",
        progress: 1,
        message: demoCancelRef.current ? "Demo stopped" : "Demo complete",
      });

      director.setSelectedHouseId(null); // re-enable auto-rotate
      director.setShowOrderFlows(false);
      director.setMaxFlows(50);
    } catch (unknownError) {
      captureError(unknownError);
      useSceneDirectorStore.getState().setDemo({
        running: false,
        phase: "Error",
        progress: 0,
        message: "Demo error",
      });
    } finally {
      setIsRunningDemo(false);
      setDemoPhase(demoCancelRef.current ? "Stopped" : "Complete");
      setDemoProgress(1);

      useSceneDirectorStore.getState().setDemo({ running: false });

      // restore polling setting
      setAutoRefresh(previousAutoRefresh);
    }
  }, [
    isRunningDemo,
    setAutoRefresh,
    resetSession,
    captureError,
    derivedWeather?.output_file_path,
    uploadedWeather?.resolved_path,
    mergeTopology,
    updateFromStep,
    setError,
  ]);

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
    isRunningDemo,
    demoPhase,
    demoProgress,
    runDemoSequence,
  };
}
