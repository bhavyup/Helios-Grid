export type PolicyMode = "rule" | "ppo-preview";
export type CsvRole = "auto" | "weather" | "household" | "market";

export interface TopologyNode {
  id: number;
  type: string;
  label: string;
}

export interface TopologyEdge {
  source: number;
  target: number;
  weight: number;
}

export interface TopologyPayload {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  node_count: number;
  edge_count: number;
}

export interface SimulationObservation {
  house_states: number[][];
  grid_state: number[][];
  market_state: number[][];
}

export interface SimulationMetrics {
  episode_id: number;
  steps_executed: number;
  cumulative_reward: number;
  average_step_reward: number;
  latest_step_reward: number;
  average_price: number;
  latest_price: number;
  peak_demand: number;
  peak_supply: number;
  total_grid_import: number;
  average_renewable_utilization: number;
}

export interface TrajectoryPoint {
  step: number;
  timestamp: string;
  reward: number;
  done: boolean;
  supply: number;
  demand: number;
  price: number;
  grid_import: number;
  renewable_utilization: number;
}

export interface SimulationStepResult {
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

export interface SimulationDataSources {
  weather_data: string;
}

export interface SimulationStateResponse {
  episode_id: number;
  seed: number;
  step: number;
  data_sources: SimulationDataSources;
  observation: SimulationObservation;
  latest_info: Record<string, unknown>;
  metrics: SimulationMetrics;
  trajectory_point: TrajectoryPoint;
  topology?: TopologyPayload;
  step_result?: SimulationStepResult;
}

export interface SimulationRunResponse {
  state: SimulationStateResponse;
  trajectory: TrajectoryPoint[];
  metrics: SimulationMetrics;
}

export interface PPORewardPoint {
  episode: number;
  reward: number;
  moving_average_reward: number;
  policy_loss: number;
  value_loss: number;
  entropy: number;
}

export interface PolicyEvaluationMetrics {
  policy_mode: string;
  episodes: number;
  steps_per_episode: number;
  average_reward: number;
  average_consumption: number;
  average_production: number;
  average_price: number;
  average_grid_import: number;
}

export interface PolicyComparison {
  ppo: PolicyEvaluationMetrics;
  rule: PolicyEvaluationMetrics;
  deltas: {
    reward_delta: number;
    grid_import_delta: number;
    price_delta: number;
  };
}

export interface TrainingRunPayload {
  run_id: string;
  created_at: string;
  training: {
    algorithm: string;
    seed: number;
    episodes: number;
    steps_per_episode: number;
    duration_seconds: number;
    reward_curve: PPORewardPoint[];
    final_training_reward: number;
    best_training_reward: number;
    final_eval_metrics: PolicyEvaluationMetrics;
  };
  comparison: PolicyComparison;
}

export interface RewardCurvePayload {
  run_id?: string;
  created_at?: string;
  reward_curve: PPORewardPoint[];
  episodes?: number;
  status?: string;
  message?: string;
}

export interface IdlePayload {
  status: "idle";
  message: string;
}

export interface CsvRoleCompatibility {
  compatible: boolean;
  matched_columns: string[];
  missing_required_all: string[];
  missing_required_any: string[];
  coverage_score: number;
  runtime_supported_now: boolean;
}

export interface CsvRoleSchema {
  description: string;
  required_all: string[];
  required_any: string[];
  recommended: string[];
  runtime_supported_now: boolean;
  runtime_usage: string;
}

export interface CsvSchemasPayload {
  weather: CsvRoleSchema;
  household: CsvRoleSchema;
  market: CsvRoleSchema;
}

export interface CsvProfilePayload {
  file_path: string;
  resolved_path: string;
  rows: number;
  column_count: number;
  columns: string[];
  numeric_columns: string[];
  null_counts: Record<string, number>;
  null_counts_scope?: string;
  preview_rows: Array<Record<string, unknown>>;
  requested_role: CsvRole;
  inferred_role: string;
  selected_role: string;
  compatibility: Record<string, CsvRoleCompatibility>;
  can_use_now: boolean;
  usage_recommendation: string;
}

export interface DerivedWeatherPayload {
  source_file_path: string;
  resolved_source_path: string;
  output_file_path: string;
  rows: number;
  columns: string[];
  column_mapping: Record<string, string | null>;
  normalization: {
    enabled: boolean;
    solar_scale: number;
    wind_scale: number;
  };
  usage_recommendation: string;
}
