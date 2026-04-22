import {
  CsvProfilePayload,
  CsvRole,
  CsvSchemasPayload,
  DerivedWeatherPayload,
  PolicyComparison,
  RewardCurvePayload,
  SimulationMetrics,
  SimulationRunResponse,
  SimulationStateResponse,
  TrajectoryPoint,
  TrainingRunPayload,
} from "@/lib/types";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/backend").replace(/\/$/, "");

class ApiError extends Error {
  readonly status: number;
  readonly payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  options?: { timeoutMs?: number },
): Promise<T> {
  const timeoutMs = options?.timeoutMs ?? 20_000;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init.headers ?? {}),
      },
      cache: "no-store",
      signal: controller.signal,
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new ApiError(
        `Request failed: ${response.status} ${response.statusText}`,
        response.status,
        payload,
      );
    }

    return payload as T;
  } finally {
    clearTimeout(timeout);
  }
}

export interface ResetSimulationInput {
  seed?: number;
  num_households?: number;
  max_episode_steps?: number;
  weather_data_path?: string;
}

export interface StepSimulationInput {
  house_actions?: number[][];
  market_action?: number;
  use_autopilot?: boolean;
}

export interface RunSimulationInput {
  steps: number;
  market_action?: number;
  use_autopilot?: boolean;
}

export interface PPOTrainingInput {
  episodes: number;
  steps_per_episode: number;
  eval_episodes: number;
  seed?: number;
  learning_rate?: number;
  hidden_dim?: number;
  clip_epsilon?: number;
}

export interface PPOComparisonInput {
  episodes: number;
  steps_per_episode: number;
  seed?: number;
}

export interface CsvProfileInput {
  file_path: string;
  role?: CsvRole;
  preview_rows?: number;
}

export interface DeriveWeatherInput {
  file_path: string;
  solar_column: string;
  wind_column: string;
  timestamp_column?: string;
  temperature_column?: string;
  humidity_column?: string;
  output_path?: string;
  normalize_signals?: boolean;
}

export const apiClient = {
  resetSimulation(input: ResetSimulationInput): Promise<SimulationStateResponse> {
    return request<SimulationStateResponse>("/simulation/reset", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },

  stepSimulation(input: StepSimulationInput): Promise<SimulationStateResponse> {
    return request<SimulationStateResponse>("/simulation/step", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },

  runSimulation(input: RunSimulationInput): Promise<SimulationRunResponse> {
    return request<SimulationRunResponse>("/simulation/run", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },

  getSimulationState(includeTopology = false): Promise<SimulationStateResponse> {
    return request<SimulationStateResponse>(
      `/simulation/state?include_topology=${includeTopology ? "true" : "false"}`,
      { method: "GET" },
    );
  },

  getSimulationMetrics(): Promise<SimulationMetrics> {
    return request<SimulationMetrics>("/simulation/metrics", { method: "GET" });
  },

  getSimulationHistory(limit = 240): Promise<TrajectoryPoint[]> {
    return request<TrajectoryPoint[]>(`/simulation/history?limit=${limit}`, { method: "GET" });
  },

  getCsvSchemas(): Promise<CsvSchemasPayload> {
    return request<CsvSchemasPayload>("/simulation/data/schemas", { method: "GET" });
  },

  profileCsvData(input: CsvProfileInput): Promise<CsvProfilePayload> {
    return request<CsvProfilePayload>("/simulation/data/profile", {
      method: "POST",
      body: JSON.stringify({
        role: "auto",
        preview_rows: 5,
        ...input,
      }),
    });
  },

  deriveWeatherCsv(input: DeriveWeatherInput): Promise<DerivedWeatherPayload> {
    return request<DerivedWeatherPayload>("/simulation/data/derive-weather", {
      method: "POST",
      body: JSON.stringify({
        normalize_signals: true,
        ...input,
      }),
    });
  },

  runPpoTraining(input: PPOTrainingInput): Promise<TrainingRunPayload> {
    return request<TrainingRunPayload>(
      "/training/ppo/run",
      {
        method: "POST",
        body: JSON.stringify(input),
      },
      { timeoutMs: 45_000 },
    );
  },

  getLatestPpoTraining(): Promise<TrainingRunPayload | { status: string; message: string }> {
    return request<TrainingRunPayload | { status: string; message: string }>("/training/ppo/latest", {
      method: "GET",
    });
  },

  comparePpoAndRule(input: PPOComparisonInput): Promise<PolicyComparison & { run_id?: string; created_at?: string }> {
    return request<PolicyComparison & { run_id?: string; created_at?: string }>("/training/ppo/compare", {
      method: "POST",
      body: JSON.stringify(input),
    });
  },

  getLatestPpoComparison(): Promise<
    (PolicyComparison & { run_id?: string; created_at?: string }) | { status: string; message: string }
  > {
    return request<(PolicyComparison & { run_id?: string; created_at?: string }) | { status: string; message: string }>(
      "/training/ppo/comparison/latest",
      {
        method: "GET",
      },
    );
  },

  getLatestRewardCurve(): Promise<RewardCurvePayload> {
    return request<RewardCurvePayload>("/training/ppo/reward-curve", {
      method: "GET",
    });
  },
};

export { ApiError };
