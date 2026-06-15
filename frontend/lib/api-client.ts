import { API_BASE_URL } from "@/lib/api-base";
import { clearAuthToken, getAuthHeader } from "@/lib/auth";
import {
  CsvProfilePayload,
  CsvRole,
  CsvPathsPayload,
  CsvSchemasPayload,
  DerivedWeatherPayload,
  UploadedWeatherPayload,
  PolicyComparison,
  RewardCurvePayload,
  SimulationMetrics,
  SimulationRunResponse,
  SimulationStateResponse,
  TrajectoryPoint,
  TrainingRunPayload,
  DerivedHouseholdPayload,
  DerivedMarketPayload,
  UploadedHouseholdPayload,
  UploadedMarketPayload,
} from "@/lib/types";

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
  options?: { timeoutMs?: number; retryOnAuth?: boolean },
): Promise<T> {
  const timeoutMs = options?.timeoutMs ?? 20_000;
  const retryOnAuth = options?.retryOnAuth ?? true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const authHeader = await getAuthHeader();

    // Normalize headers to a Headers instance to satisfy HeadersInit typing
    const headers = new Headers(init.headers as HeadersInit | undefined);
    const isFormDataBody = init.body instanceof FormData;
    if (!isFormDataBody) {
      headers.set("Content-Type", "application/json");
    }
    if (authHeader && typeof authHeader === "object") {
      for (const [k, v] of Object.entries(authHeader)) {
        if (v !== undefined && v !== null) {
          headers.set(k, String(v));
        }
      }
    }

    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      cache: "no-store",
      signal: controller.signal,
    });

    if (response.status === 401 && retryOnAuth) {
      clearAuthToken();
      await getAuthHeader({ forceRefresh: true });
      return request<T>(path, init, { ...options, retryOnAuth: false });
    }

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401) {
        clearAuthToken();
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
      }
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
  household_data_path?: string;
  market_data_path?: string;
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
  wait_for_result?: boolean;
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
  irradiance_column?: string;
  ghi_column?: string;
  dni_column?: string;
  dhi_column?: string;
  pv_power_column?: string;
  output_path?: string;
  normalize_signals?: boolean;
  panel_tilt?: number;
  panel_azimuth?: number;
  panel_area?: number;
  panel_efficiency?: number;
  temp_coefficient?: number;
}

export interface UploadWeatherInput {
  file: File;
}

export interface DeriveHouseholdInput {
  file_path: string;
  consumption_column: string;
  timestamp_column?: string;
  household_id_column?: string;
  pv_generation_column?: string;
  net_load_column?: string;
  output_path?: string;
  normalize_signals?: boolean;
}

export interface DeriveMarketInput {
  file_path: string;
  supply_column?: string;
  demand_column?: string;
  price_column: string;
  timestamp_column?: string;
  bid_column?: string;
  ask_column?: string;
  clearing_price_column?: string;
  output_path?: string;
  normalize_signals?: boolean;
}

export interface UploadHouseholdInput {
  file: File;
}

export interface UploadMarketInput {
  file: File;
}

export const apiClient = {
  resetSimulation(
    input: ResetSimulationInput,
  ): Promise<SimulationStateResponse> {
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

  getSimulationState(
    includeTopology = false,
  ): Promise<SimulationStateResponse> {
    return request<SimulationStateResponse>(
      `/simulation/state?include_topology=${includeTopology ? "true" : "false"}`,
      { method: "GET" },
    );
  },

  getSimulationMetrics(): Promise<SimulationMetrics> {
    return request<SimulationMetrics>("/simulation/metrics", { method: "GET" });
  },

  getSimulationHistory(limit = 240): Promise<TrajectoryPoint[]> {
    return request<TrajectoryPoint[]>(`/simulation/history?limit=${limit}`, {
      method: "GET",
    });
  },

  getCsvSchemas(): Promise<CsvSchemasPayload> {
    return request<CsvSchemasPayload>("/simulation/data/schemas", {
      method: "GET",
    });
  },

  getCsvPaths(): Promise<CsvPathsPayload> {
    return request<CsvPathsPayload>("/simulation/data/paths", {
      method: "GET",
    });
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

  uploadWeatherCsv(input: UploadWeatherInput): Promise<UploadedWeatherPayload> {
    const formData = new FormData();
    formData.append("file", input.file);

    return request<UploadedWeatherPayload>(
      "/simulation/data/upload-weather",
      {
        method: "POST",
        body: formData,
      },
      { timeoutMs: 120_000 },
    );
  },

  deriveHouseholdCsv(
    input: DeriveHouseholdInput,
  ): Promise<DerivedHouseholdPayload> {
    return request<DerivedHouseholdPayload>(
      "/simulation/data/derive-household",
      {
        method: "POST",
        body: JSON.stringify({
          normalize_signals: false,
          ...input,
        }),
      },
    );
  },

  deriveMarketCsv(input: DeriveMarketInput): Promise<DerivedMarketPayload> {
    return request<DerivedMarketPayload>("/simulation/data/derive-market", {
      method: "POST",
      body: JSON.stringify({
        normalize_signals: false,
        ...input,
      }),
    });
  },

  uploadHouseholdCsv(
    input: UploadHouseholdInput,
  ): Promise<UploadedHouseholdPayload> {
    const formData = new FormData();
    formData.append("file", input.file);

    return request<UploadedHouseholdPayload>(
      "/simulation/data/upload-household",
      { method: "POST", body: formData },
      { timeoutMs: 120_000 },
    );
  },

  uploadMarketCsv(input: UploadMarketInput): Promise<UploadedMarketPayload> {
    const formData = new FormData();
    formData.append("file", input.file);

    return request<UploadedMarketPayload>(
      "/simulation/data/upload-market",
      { method: "POST", body: formData },
      { timeoutMs: 120_000 },
    );
  },

  runPpoTraining(input: PPOTrainingInput): Promise<TrainingRunPayload> {
    return request<TrainingRunPayload>(
      "/training/ppo/run",
      {
        method: "POST",
        body: JSON.stringify(input),
      },
      { timeoutMs: 120_000 },
    );
  },

  getLatestPpoTraining(): Promise<
    TrainingRunPayload | { status: string; message: string }
  > {
    return request<TrainingRunPayload | { status: string; message: string }>(
      "/training/ppo/latest",
      {
        method: "GET",
      },
    );
  },

  comparePpoAndRule(
    input: PPOComparisonInput,
  ): Promise<PolicyComparison & { run_id?: string; created_at?: string }> {
    return request<PolicyComparison & { run_id?: string; created_at?: string }>(
      "/training/ppo/compare",
      {
        method: "POST",
        body: JSON.stringify(input),
      },
    );
  },

  getLatestPpoComparison(): Promise<
    | (PolicyComparison & { run_id?: string; created_at?: string })
    | { status: string; message: string }
  > {
    return request<
      | (PolicyComparison & { run_id?: string; created_at?: string })
      | { status: string; message: string }
    >("/training/ppo/comparison/latest", {
      method: "GET",
    });
  },

  getLatestRewardCurve(): Promise<RewardCurvePayload> {
    return request<RewardCurvePayload>("/training/ppo/reward-curve", {
      method: "GET",
    });
  },
};

export { ApiError };
