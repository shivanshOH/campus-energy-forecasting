export type MeasurementConfig = {
  source_unit: "W" | "kW" | "Wh" | "kWh";
  semantics: "instantaneous_power" | "interval_energy" | "cumulative_energy";
  target_unit: "Wh" | "kWh";
  source_column?: string;
  expected_interval_minutes?: number;
  irregular_interval_policy: "use_actual_interval" | "reject_gaps" | "clip_to_expected";
  gap_factor: number;
  last_interval_policy: "drop" | "expected_duration";
};

export const defaultMeasurement: MeasurementConfig = {
  source_unit: "W",
  semantics: "instantaneous_power",
  target_unit: "Wh",
  expected_interval_minutes: 10,
  irregular_interval_policy: "use_actual_interval",
  gap_factor: 1.5,
  last_interval_policy: "drop",
};

export type Dataset = {
  dataset_id: string;
  identifier: string;
  version: string;
  root_path: string;
  audit?: Record<string, any>;
};

export type Experiment = {
  experiment_id: string;
  dataset_id: string;
  model_name: string;
  status: string;
  building_id?: string;
  feature_configuration: string;
  metrics: Record<string, number | null>;
  metadata?: Record<string, any>;
};

export type Building = {
  building_id: string;
  name: string;
  category: string;
  datasets: string[];
};

export type ForecastPoint = {
  experiment_id: string;
  building_id: string;
  issue_timestamp: string;
  horizon_step: number;
  p10?: number | null;
  p50?: number | null;
  p90?: number | null;
  actual?: number | null;
};

export type Job = {
  job_id: string;
  status: string;
  result?: any;
  error?: string;
};

export type ScenarioResult = {
  label: string;
  interpretation: string;
  baseline: number[][];
  scenario: number[][];
  difference: number[][];
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options?.headers ?? {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail);
      detail = parsed.detail ?? detail;
    } catch {
      // Preserve plain-text server errors.
    }
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export const api = {
  health: () => request<{ status: string; environment: string; version: string }>("/api/health"),
  datasets: () => request<Dataset[]>("/api/datasets"),
  buildings: () => request<Building[]>("/api/buildings"),
  experiments: () => request<Experiment[]>("/api/experiments"),
  experiment: (id: string) => request<Experiment>(`/api/experiments/${id}`),
  forecasts: (id: string) => request<ForecastPoint[]>(`/api/experiments/${id}/forecasts`),
  explanation: (id: string) => request<any>(`/api/explanations/${id}`),
  job: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  audit: (rootPath: string, measurement: MeasurementConfig) =>
    post<Record<string, any>>(`/api/datasets/audit`, {
      root_path: rootPath,
      dataset_id: "i_blend",
      dataset_version: "local",
      energy_measurement: measurement,
    }),
  energyPredict: (body: {
    building_id: string;
    horizon: string;
    model_name: string;
    feature_configuration: string;
    occupancy_mode: string;
    occupancy_model_name: string;
    diagnostic?: boolean;
    energy_measurement: MeasurementConfig;
  }) => post<{ job_id: string; status: string; message: string }>("/api/energy/predict", {
    dataset_id: "i_blend",
    ...body,
  }),
  occupancyTrain: (body: {
    building_id: string;
    horizon: string;
    model_name: string;
    occupancy_model_name: string;
  }) => post<{ job_id: string; status: string; message: string }>("/api/occupancy/train", {
    dataset_id: "i_blend",
    feature_configuration: "E_predicted_occupancy",
    ...body,
  }),
  robustness: (body: Record<string, any>) => post<{ job_id: string; status: string; message: string }>("/api/experiments/robustness", {
    dataset_id: "i_blend",
    energy_measurement: defaultMeasurement,
    ...body,
  }),
  transfer: (body: Record<string, any>) => post<{ job_id: string; status: string; message: string }>("/api/experiments/transfer", {
    dataset_id: "i_blend",
    energy_measurement: defaultMeasurement,
    ...body,
  }),
  report: (experimentId: string, title: string) => post<{ path: string }>("/api/reports/generate", {
    experiment_id: experimentId,
    title,
    format: "html",
  }),
  scenario: (body: {
    experiment_id: string;
    occupancy_multiplier: number;
    temperature_delta_c: number;
    humidity_delta_pct: number;
  }) => post<ScenarioResult>("/api/scenarios", body),
};
