import { ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  Building,
  Dataset,
  defaultMeasurement,
  Experiment,
  ForecastPoint,
  MeasurementConfig,
  ScenarioResult,
} from "./api";

type Page =
  | "overview"
  | "dataset"
  | "occupancy"
  | "energy"
  | "transfer"
  | "robustness"
  | "models"
  | "explainability"
  | "scenario"
  | "reports";

const navigation: { id: Page; label: string; icon: string }[] = [
  { id: "overview", label: "Overview", icon: "⌂" },
  { id: "dataset", label: "Dataset explorer", icon: "▦" },
  { id: "occupancy", label: "Occupancy forecasting", icon: "◌" },
  { id: "energy", label: "Energy forecasting", icon: "ϟ" },
  { id: "transfer", label: "Cross-building", icon: "⇄" },
  { id: "robustness", label: "Robustness", icon: "⊙" },
  { id: "models", label: "Model comparison", icon: "▥" },
  { id: "explainability", label: "Explainability", icon: "✦" },
  { id: "scenario", label: "Scenario simulator", icon: "◈" },
  { id: "reports", label: "Research reports", icon: "≡" },
];

function App() {
  const [page, setPage] = useState<Page>("overview");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [buildings, setBuildings] = useState<Building[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [measurement, setMeasurement] = useState<MeasurementConfig>(defaultMeasurement);
  const [health, setHealth] = useState("checking");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [healthResponse, datasetResponse, buildingResponse, experimentResponse] =
        await Promise.all([api.health(), api.datasets(), api.buildings(), api.experiments()]);
      setHealth(healthResponse.status);
      setDatasets(datasetResponse);
      setBuildings(buildingResponse);
      setExperiments(experimentResponse);
      setError(null);
    } catch (reason) {
      setHealth("offline");
      setError(reason instanceof Error ? reason.message : "Backend unavailable");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">CE</div>
          <div>
            <div className="brand-name">Campus Energy</div>
            <div className="brand-subtitle">Research intelligence</div>
          </div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? "active" : ""}`}
              onClick={() => setPage(item.id)}
            >
              <span className="nav-icon" aria-hidden="true">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className={`status-dot ${health === "ok" ? "online" : ""}`} />
          <div>
            <strong>API {health}</strong>
            <span>Local research runtime</span>
          </div>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">RESEARCH PLATFORM / 2026</div>
            <h1>{navigation.find((item) => item.id === page)?.label}</h1>
          </div>
          <div className="topbar-actions">
            <span className="integrity-badge">
              <span className="status-dot online" /> Leakage-safe mode
            </span>
            <button className="icon-button" onClick={() => void refresh()} aria-label="Refresh data">
              ↻
            </button>
          </div>
        </header>
        {error && <div className="alert">Backend data is unavailable: {error}</div>}
        {page === "overview" && <Overview buildings={buildings} datasets={datasets} experiments={experiments} />}
        {page === "dataset" && <DatasetExplorer datasets={datasets} measurement={measurement} onMeasurementChange={setMeasurement} onRefresh={refresh} />}
        {page === "occupancy" && <OccupancyPage buildings={buildings} experiments={experiments} onRefresh={refresh} />}
        {page === "energy" && <EnergyPage buildings={buildings} experiments={experiments} measurement={measurement} onMeasurementChange={setMeasurement} onRefresh={refresh} />}
        {page === "transfer" && <TransferPage buildings={buildings} experiments={experiments} measurement={measurement} onMeasurementChange={setMeasurement} onRefresh={refresh} />}
        {page === "robustness" && <RobustnessPage experiments={experiments} measurement={measurement} onMeasurementChange={setMeasurement} onRefresh={refresh} />}
        {page === "models" && <ModelPage experiments={experiments} />}
        {page === "explainability" && <ExplainabilityPage experiments={experiments} />}
        {page === "scenario" && <ScenarioPage experiments={experiments} />}
        {page === "reports" && <ReportsPage experiments={experiments} />}
      </main>
    </div>
  );
}

function StatCard({ label, value, detail, tone = "blue" }: { label: string; value: string; detail: string; tone?: string }) {
  return (
    <div className="stat-card">
      <div className={`stat-accent ${tone}`} />
      <span className="card-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      <span className="stat-detail">{detail}</span>
    </div>
  );
}

function SectionHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="section-header">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

function EmptyState({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

function MeasurementEditor({
  value,
  onChange,
}: {
  value: MeasurementConfig;
  onChange: (value: MeasurementConfig) => void;
}) {
  const update = <K extends keyof MeasurementConfig>(key: K, next: MeasurementConfig[K]) =>
    onChange({ ...value, [key]: next });
  return (
    <div className="measurement-editor">
      <div className="measurement-title">Explicit energy measurement semantics</div>
      <label>Source unit<select value={value.source_unit} onChange={(event) => update("source_unit", event.target.value as MeasurementConfig["source_unit"])}><option value="W">W</option><option value="kW">kW</option><option value="Wh">Wh</option><option value="kWh">kWh</option></select></label>
      <label>Semantics<select value={value.semantics} onChange={(event) => update("semantics", event.target.value as MeasurementConfig["semantics"])}><option value="instantaneous_power">Instantaneous power</option><option value="interval_energy">Interval energy</option><option value="cumulative_energy">Cumulative energy</option></select></label>
      <label>Target unit<select value={value.target_unit} onChange={(event) => update("target_unit", event.target.value as MeasurementConfig["target_unit"])}><option value="Wh">Wh</option><option value="kWh">kWh</option></select></label>
      <label>Source column<input value={value.source_column ?? ""} placeholder="Optional exact label" onChange={(event) => update("source_column", event.target.value || undefined)} /></label>
      <label>Expected minutes<input type="number" min="0.001" step="0.001" value={value.expected_interval_minutes ?? ""} onChange={(event) => update("expected_interval_minutes", event.target.value ? Number(event.target.value) : undefined)} /></label>
      <label>Irregular intervals<select value={value.irregular_interval_policy} onChange={(event) => update("irregular_interval_policy", event.target.value as MeasurementConfig["irregular_interval_policy"])}><option value="use_actual_interval">Use actual duration</option><option value="reject_gaps">Reject large gaps</option><option value="clip_to_expected">Clip large gaps</option></select></label>
      <label>Final interval<select value={value.last_interval_policy} onChange={(event) => update("last_interval_policy", event.target.value as MeasurementConfig["last_interval_policy"])}><option value="drop">Drop</option><option value="expected_duration">Use expected duration</option></select></label>
    </div>
  );
}

function useForecastData(experimentId: string | undefined) {
  const [forecasts, setForecasts] = useState<ForecastPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!experimentId) {
      setForecasts([]);
      return;
    }
    setLoading(true);
    void api
      .forecasts(experimentId)
      .then(setForecasts)
      .catch((reason) => setError(reason instanceof Error ? reason.message : "Forecast request failed"))
      .finally(() => setLoading(false));
  }, [experimentId]);
  return { forecasts, loading, error };
}

function ForecastChart({ forecasts, title = "Persisted forecast" }: { forecasts: ForecastPoint[]; title?: string }) {
  const chartData = useMemo(() => {
    if (!forecasts.length) return [];
    const latestIssue = forecasts.reduce((latest, point) =>
      point.issue_timestamp > latest ? point.issue_timestamp : latest,
      forecasts[0].issue_timestamp,
    );
    return forecasts
      .filter((point) => point.issue_timestamp === latestIssue)
      .map((point) => ({
        horizon: `h${point.horizon_step}`,
        actual: point.actual ?? undefined,
        p10: point.p10 ?? undefined,
        p50: point.p50 ?? undefined,
        p90: point.p90 ?? undefined,
      }));
  }, [forecasts]);
  return (
    <div className="panel chart-panel">
      <SectionHeader title={title} subtitle={forecasts.length ? "Latest issue time from persisted results" : "No persisted forecast selected"} />
      {chartData.length ? (
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#253149" vertical={false} />
              <XAxis dataKey="horizon" stroke="#73819a" tickLine={false} axisLine={false} />
              <YAxis stroke="#73819a" tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: "#111b2d", border: "1px solid #2a3852" }} />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#d3dcea" dot name="Actual" />
              <Line type="monotone" dataKey="p50" stroke="#5da9ff" strokeWidth={3} dot={false} name="P50" />
              <Line type="monotone" dataKey="p10" stroke="#ae94ff" strokeDasharray="4 4" dot={false} name="P10" />
              <Line type="monotone" dataKey="p90" stroke="#41d5bb" strokeDasharray="4 4" dot={false} name="P90" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <EmptyState>Run a real experiment to populate this chart. No static values are shown.</EmptyState>
      )}
    </div>
  );
}

function pollJob(jobId: string, setStatus: (value: string) => void, onComplete: (job: any) => void) {
  let cancelled = false;
  const poll = async () => {
    try {
      const job = await api.job(jobId);
      if (cancelled) return;
      setStatus(job.status === "running" ? "Experiment running…" : job.status);
      if (job.status === "completed" || job.status === "failed") {
        onComplete(job);
        return;
      }
      window.setTimeout(() => void poll(), 800);
    } catch (reason) {
      if (!cancelled) setStatus(reason instanceof Error ? reason.message : "Job polling failed");
    }
  };
  void poll();
  return () => {
    cancelled = true;
  };
}

function Overview({ buildings, datasets, experiments }: { buildings: Building[]; datasets: Dataset[]; experiments: Experiment[] }) {
  const latest = experiments.find((experiment) => !experiment.model_name.endsWith("_occupancy")) ?? experiments[0];
  const { forecasts } = useForecastData(latest?.experiment_id);
  const metricRows = experiments.slice(0, 6).map((experiment) => ({
    model: experiment.model_name.replaceAll("_", " "),
    mae: experiment.metrics["horizon_1_mae"],
    r2: experiment.metrics["horizon_1_r2"],
  }));
  return (
    <div className="page-stack">
      <div className="hero-panel">
        <div>
          <span className="eyebrow">OCCUPANCY-AWARE FORECASTING</span>
          <h2>Make every forecast explainable.</h2>
          <p>One research workspace for data quality, multi-horizon energy prediction, uncertainty and cross-building transfer.</p>
        </div>
        <div className="hero-metric"><span>ACTIVE HORIZON</span><strong>{latest ? `${latest.metadata?.forecast_horizon_steps ?? "—"}` : "—"}</strong><small>Persisted result steps</small></div>
      </div>
      <div className="stats-grid">
        <StatCard label="Buildings analysed" value={String(buildings.length || "—")} detail="From registered audit coverage" />
        <StatCard label="Experiments" value={String(experiments.length || "—")} detail="Persisted actual runs" tone="teal" />
        <StatCard label="Datasets" value={String(datasets.length || "—")} detail="Audited local sources" tone="purple" />
        <StatCard label="Research status" value={experiments.length ? "Active" : "Ready"} detail="No fabricated results" tone="orange" />
      </div>
      <div className="two-column-grid">
        <ForecastChart forecasts={forecasts} title="Latest persisted forecast" />
        <div className="panel"><SectionHeader title="Recent model performance" subtitle="Actual persisted metrics" /><div className="table-wrap"><table><thead><tr><th>Model</th><th>MAE</th><th>R²</th></tr></thead><tbody>{metricRows.length ? metricRows.map((row, index) => <tr key={`${row.model}-${index}`}><td className="capitalize">{row.model}</td><td>{row.mae == null ? "—" : row.mae.toFixed(3)}</td><td>{row.r2 == null ? "—" : row.r2.toFixed(3)}</td></tr>) : <tr><td colSpan={3} className="empty-cell">No completed experiments yet.</td></tr>}</tbody></table></div></div>
      </div>
    </div>
  );
}

function DatasetExplorer({ datasets, measurement, onMeasurementChange, onRefresh }: { datasets: Dataset[]; measurement: MeasurementConfig; onMeasurementChange: (value: MeasurementConfig) => void; onRefresh: () => Promise<void> }) {
  const [rootPath, setRootPath] = useState("data/raw/i_blend");
  const [audit, setAudit] = useState<Record<string, any> | null>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const dataset = datasets[0];
  const coverage = (dataset?.audit?.coverage_by_kind ?? {}) as Record<string, { start?: string; end?: string; rows?: number }>;
  const runAudit = async () => {
    setLoading(true); setStatus("Auditing source files…");
    try { const result = await api.audit(rootPath, measurement); setAudit(result); setStatus(result.research_ready ? "Audit complete: research-ready unit semantics supplied." : "Audit complete: energy semantics still need resolution."); await onRefresh(); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Audit failed"); } finally { setLoading(false); }
  };
  return <div className="page-stack"><SectionHeader title="Dataset explorer" subtitle="Audit source coverage and physical units before training." /><div className="control-panel"><label>Dataset root<input value={rootPath} onChange={(event) => setRootPath(event.target.value)} /></label><button className="primary-button" disabled={loading} onClick={() => void runAudit()}>{loading ? "Auditing…" : "Audit dataset"}</button></div><MeasurementEditor value={measurement} onChange={onMeasurementChange} /><div className="notice-line">{status}</div><div className="stats-grid small"><StatCard label="Energy rows" value={coverage.energy?.rows?.toLocaleString() ?? "—"} detail={coverage.energy?.start ? `${coverage.energy.start.slice(0, 10)} → ${coverage.energy.end?.slice(0, 10)}` : "Audit required"} /><StatCard label="Occupancy rows" value={coverage.occupancy?.rows?.toLocaleString() ?? "—"} detail="Ten-minute source coverage" tone="teal" /><StatCard label="Weather rows" value={coverage.weather?.rows?.toLocaleString() ?? "—"} detail="Observed overlap only" tone="orange" /></div><div className="two-column-grid"><div className="panel"><SectionHeader title="Source coverage" /><div className="coverage-list">{Object.entries(coverage).map(([kind, value]) => <div className="coverage-row" key={kind}><span className="coverage-kind">{kind}</span><span>{value.rows?.toLocaleString() ?? "—"} rows</span><span>{value.start?.slice(0, 10) ?? "—"} → {value.end?.slice(0, 10) ?? "—"}</span></div>)}{!Object.keys(coverage).length && <EmptyState>Audit a licensed I-BLEND directory to populate this view.</EmptyState>}</div></div><div className="panel"><SectionHeader title="Unit readiness" /><p className="chart-note">The selected source unit and measurement semantics are sent with every energy operation. No unit is inferred from magnitude.</p><div className="policy-card"><strong>{audit?.research_ready ?? dataset?.audit?.research_ready ? "Research-ready" : "Awaiting explicit unit audit"}</strong><span>Never inferred from magnitude</span></div></div></div></div>;
}

function OccupancyPage({ buildings, experiments, onRefresh }: { buildings: Building[]; experiments: Experiment[]; onRefresh: () => Promise<void> }) {
  const [building, setBuilding] = useState(buildings[0]?.building_id ?? "");
  const [model, setModel] = useState("seasonal_naive");
  const [status, setStatus] = useState("Select a building and run an occupancy experiment.");
  const occupancyExperiments = experiments.filter((item) => item.model_name.endsWith("_occupancy"));
  const selected = occupancyExperiments[0];
  useEffect(() => { if (!building && buildings[0]) setBuilding(buildings[0].building_id); }, [building, buildings]);
  const preview = (selected?.metadata?.model_architecture?.forecast_preview ?? []) as any[];
  const chartData = preview.flatMap((row) => row.horizons ?? []).slice(0, 100).map((point) => ({ horizon: `h${point.horizon_step}`, actual: point.actual, predicted: point.predicted }));
  const run = async () => {
    if (!building) { setStatus("Register a dataset and building first."); return; }
    setStatus("Queueing occupancy experiment…");
    try { const job = await api.occupancyTrain({ building_id: building, horizon: "1h", model_name: model, occupancy_model_name: model }); pollJob(job.job_id, setStatus, async (done) => { setStatus(done.status === "completed" ? "Occupancy experiment completed." : done.error ?? "Occupancy experiment failed"); if (done.status === "completed") await onRefresh(); }); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Occupancy request failed"); }
  };
  return <div className="page-stack"><SectionHeader title="Occupancy forecasting" subtitle="Forecast occupancy first; use only out-of-sample issue-time predictions in energy experiments." action={<span className="soft-badge teal">Issue-time bounded</span>} /><div className="control-panel"><label>Building<select value={building} onChange={(event) => setBuilding(event.target.value)}><option value="">Select building</option>{buildings.map((item) => <option key={item.building_id} value={item.building_id}>{item.name}</option>)}</select></label><label>Model<select value={model} onChange={(event) => setModel(event.target.value)}><option value="seasonal_naive">Seasonal naive</option><option value="tree">HistGradientBoosting</option></select></label><button className="primary-button" onClick={() => void run()}>Run occupancy experiment →</button></div><div className="notice-line">{status}</div><div className="two-column-grid"><div className="panel chart-panel"><SectionHeader title="Observed vs predicted occupancy" subtitle={selected ? `${selected.model_name} · persisted preview` : "No occupancy experiment selected"} />{chartData.length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke="#253149" vertical={false} /><XAxis dataKey="horizon" stroke="#73819a" /><YAxis stroke="#73819a" /><Tooltip contentStyle={{ background: "#111b2d", border: "1px solid #2a3852" }} /><Legend /><Line dataKey="actual" stroke="#d3dcea" name="Actual" /><Line dataKey="predicted" stroke="#41d5bb" name="Predicted" /></LineChart></ResponsiveContainer></div> : <EmptyState>No persisted occupancy results yet.</EmptyState>}</div><div className="panel"><SectionHeader title="Model policy" /><div className="policy-card"><strong>Operational forecast</strong><span>Predicted values only</span><p>Actual future occupancy is never supplied to the operational energy configuration. Oracle occupancy remains a separate diagnostic.</p></div><div className="mini-list"><span>Buildings in scope</span><strong>{buildings.length || "—"}</strong></div></div></div></div>;
}

function EnergyPage({ buildings, experiments, measurement, onMeasurementChange, onRefresh }: { buildings: Building[]; experiments: Experiment[]; measurement: MeasurementConfig; onMeasurementChange: (value: MeasurementConfig) => void; onRefresh: () => Promise<void> }) {
  const [building, setBuilding] = useState(buildings[0]?.building_id ?? "");
  const [horizon, setHorizon] = useState("1h");
  const [model, setModel] = useState("random_forest");
  const [status, setStatus] = useState("Select a building and generate a forecast.");
  const energyExperiments = experiments.filter((item) => !item.model_name.endsWith("_occupancy"));
  const selected = energyExperiments[0];
  useEffect(() => { if (!building && buildings[0]) setBuilding(buildings[0].building_id); }, [building, buildings]);
  const { forecasts, loading, error: forecastError } = useForecastData(selected?.experiment_id);
  const generate = async () => {
    if (!building) { setStatus("Register a dataset and building first."); return; }
    setStatus("Queueing energy experiment…");
    try { const job = await api.energyPredict({ building_id: building, horizon, model_name: model, feature_configuration: "E_predicted_occupancy", occupancy_mode: "predicted", occupancy_model_name: "seasonal_naive", energy_measurement: measurement }); pollJob(job.job_id, setStatus, async (done) => { setStatus(done.status === "completed" ? "Forecast completed and persisted." : done.error ?? "Forecast failed"); if (done.status === "completed") await onRefresh(); }); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Forecast request failed"); }
  };
  return <div className="page-stack"><SectionHeader title="Energy forecasting" subtitle="Direct multi-horizon forecasts with explicit source-unit semantics." /><div className="control-panel"><label>Building<select value={building} onChange={(event) => setBuilding(event.target.value)}><option value="">Select building</option>{buildings.map((item) => <option key={item.building_id} value={item.building_id}>{item.name}</option>)}</select></label><label>Horizon<select value={horizon} onChange={(event) => setHorizon(event.target.value)}><option value="10m">10 minutes</option><option value="1h">1 hour</option><option value="6h">6 hours</option><option value="24h">24 hours</option></select></label><label>Model<select value={model} onChange={(event) => setModel(event.target.value)}><option value="random_forest">Random Forest</option><option value="linear_regression">Linear Regression</option><option value="decode_lstm">DECODE-style LSTM</option><option value="quantile_direct">Quantile + conformal</option></select></label><button className="primary-button" onClick={() => void generate()}>Generate forecast →</button></div><MeasurementEditor value={measurement} onChange={onMeasurementChange} /><div className="notice-line">{status}</div>{loading && <div className="notice-line">Loading persisted forecast points…</div>}{forecastError && <div className="alert">{forecastError}</div>}<div className="two-column-grid"><ForecastChart forecasts={forecasts} title={selected ? `${selected.model_name} · ${selected.metadata?.target_unit ?? "target unit recorded"}` : "Forecast trajectory"} /><div className="panel"><SectionHeader title="Forecast metadata" /><div className="interval-display"><span>Target unit</span><strong>{selected?.metadata?.target_unit ?? "—"}</strong><div><span>Measurement source</span><span>{selected?.metadata?.measurement_metadata?.energy_measurement_config?.source_unit ?? "—"}</span></div><div><span>Occupancy model</span><span>{selected?.metadata?.measurement_metadata?.occupancy_model ?? "—"}</span></div></div><div className="chart-note">No values are shown until a real backend experiment returns them.</div></div></div></div>;
}

function TransferPage({ buildings, experiments, measurement, onMeasurementChange, onRefresh }: { buildings: Building[]; experiments: Experiment[]; measurement: MeasurementConfig; onMeasurementChange: (value: MeasurementConfig) => void; onRefresh: () => Promise<void> }) {
  const [target, setTarget] = useState(buildings[0]?.building_id ?? "");
  const [status, setStatus] = useState("Launch a leave-one-building-out experiment.");
  const transfer = experiments.find((item) => item.model_name.startsWith("transfer_"));
  useEffect(() => { if (!target && buildings[0]) setTarget(buildings[0].building_id); }, [target, buildings]);
  const run = async () => { if (!target) { setStatus("Select a target building."); return; } setStatus("Queueing transfer experiment…"); try { const job = await api.transfer({ target_building: target, model_name: "random_forest", horizon: "1h", lookback_hours: 24, feature_configuration: "B_energy_calendar", random_seed: 42, adaptation_days: [1, 3, 7, 14, 30], energy_measurement: measurement }); pollJob(job.job_id, setStatus, async (done) => { setStatus(done.status === "completed" ? "Transfer experiment completed." : done.error ?? "Transfer experiment failed"); if (done.status === "completed") await onRefresh(); }); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Transfer request failed"); } };
  return <div className="page-stack"><SectionHeader title="Cross-building generalization" subtitle="Pooled zero-shot inference and few-shot target-building adaptation." /><div className="control-panel"><label>Unseen target<select value={target} onChange={(event) => setTarget(event.target.value)}><option value="">Select target</option>{buildings.map((item) => <option key={item.building_id} value={item.building_id}>{item.name}</option>)}</select></label><button className="primary-button" onClick={() => void run()}>Run transfer experiment →</button></div><MeasurementEditor value={measurement} onChange={onMeasurementChange} /><div className="notice-line">{status}</div><div className="stats-grid small"><StatCard label="Zero-shot folds" value="7" detail="Planned for I-BLEND buildings" /><StatCard label="Adaptation budgets" value="5" detail="1 / 3 / 7 / 14 / 30 days" tone="teal" /><StatCard label="Saved transfer runs" value={transfer ? "1" : "—"} detail="Actual results only" tone="purple" /></div><div className="panel"><SectionHeader title="Transferability result" subtitle="Persisted transfer metadata" />{transfer ? <pre className="result-pre">{JSON.stringify(transfer.metadata?.model_architecture?.transferability_matrix ?? {}, null, 2)}</pre> : <EmptyState>Awaiting a real transfer experiment.</EmptyState>}</div></div>;
}

function RobustnessPage({ experiments, measurement, onMeasurementChange, onRefresh }: { experiments: Experiment[]; measurement: MeasurementConfig; onMeasurementChange: (value: MeasurementConfig) => void; onRefresh: () => Promise<void> }) {
  const [occupancyMissing, setOccupancyMissing] = useState(10);
  const [weatherMissing, setWeatherMissing] = useState(10);
  const [outage, setOutage] = useState(0);
  const [status, setStatus] = useState("Configure a controlled corruption experiment.");
  const robustness = experiments.find((item) => item.model_name.startsWith("robustness_"));
  const run = async () => { setStatus("Queueing robustness experiment…"); try { const job = await api.robustness({ building_id: undefined, model_name: "random_forest", horizon: "1h", lookback_hours: 24, feature_configuration: "B_energy_calendar", random_seed: 42, occupancy_missing_percent: occupancyMissing, weather_missing_percent: weatherMissing, outage_hours: outage, imputation_strategy: "causal", energy_measurement: measurement }); pollJob(job.job_id, setStatus, async (done) => { setStatus(done.status === "completed" ? "Robustness experiment completed." : done.error ?? "Robustness experiment failed"); if (done.status === "completed") await onRefresh(); }); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Robustness request failed"); } };
  return <div className="page-stack"><SectionHeader title="Robustness analysis" subtitle="Controlled sensor failures, corruption seeds and causal imputation." /><div className="control-panel"><label>Occupancy missingness<select value={occupancyMissing} onChange={(event) => setOccupancyMissing(Number(event.target.value))}><option value={5}>5%</option><option value={10}>10%</option><option value={20}>20%</option><option value={30}>30%</option></select></label><label>Weather missingness<select value={weatherMissing} onChange={(event) => setWeatherMissing(Number(event.target.value))}><option value={5}>5%</option><option value={10}>10%</option><option value={20}>20%</option><option value={30}>30%</option></select></label><label>Outage duration<select value={outage} onChange={(event) => setOutage(Number(event.target.value))}><option value={0}>Random missingness</option><option value={6}>6 hours</option><option value={24}>24 hours</option></select></label><button className="secondary-button" onClick={() => void run()}>Run robustness experiment</button></div><MeasurementEditor value={measurement} onChange={onMeasurementChange} /><div className="notice-line">{status}</div><div className="panel"><SectionHeader title="Performance degradation" subtitle="Clean and corrupted metrics from the same persisted test period" />{robustness ? <div className="table-wrap"><table><tbody>{Object.entries(robustness.metrics).filter(([key]) => key.includes("degradation") || key.endsWith("mae")).slice(0, 12).map(([key, value]) => <tr key={key}><td>{key}</td><td>{value == null ? "—" : Number(value).toFixed(4)}</td></tr>)}</tbody></table></div> : <EmptyState>No robustness experiment result yet.</EmptyState>}</div></div>;
}

function ModelPage({ experiments }: { experiments: Experiment[] }) { const data = experiments.slice(0, 8).map((item) => ({ name: item.model_name, mae: item.metrics["horizon_1_mae"] })).filter((item): item is { name: string; mae: number } => typeof item.mae === "number"); return <div className="page-stack"><SectionHeader title="Model comparison" subtitle="Metrics are populated only from persisted experiments." /><div className="panel chart-panel">{data.length ? <div className="chart-wrap large"><ResponsiveContainer width="100%" height="100%"><BarChart data={data}><CartesianGrid strokeDasharray="3 3" stroke="#253149" vertical={false} /><XAxis dataKey="name" stroke="#73819a" /><YAxis stroke="#73819a" /><Tooltip contentStyle={{ background: "#111b2d", border: "1px solid #2a3852" }} /><Bar dataKey="mae" fill="#5da9ff" radius={[4, 4, 0, 0]} name="MAE" /></BarChart></ResponsiveContainer></div> : <EmptyState>Run a real model experiment to compare metrics.</EmptyState>}</div><div className="panel"><SectionHeader title="Comparison suite" /><div className="tag-grid">{["Persistence", "Seasonal naive", "Linear regression", "Decision tree", "Random forest", "DECODE LSTM", "LightGBM", "GRU"].map((model) => <span className="model-tag" key={model}>{model}<small>{experiments.some((item) => item.model_name.toLowerCase().includes(model.toLowerCase().replaceAll(" ", "_"))) ? "measured" : "not run"}</small></span>)}</div></div></div>; }

function ExplainabilityPage({ experiments }: { experiments: Experiment[] }) { const [selected, setSelected] = useState(experiments[0]?.experiment_id ?? ""); const [result, setResult] = useState<any>(null); const load = async () => { if (selected) setResult(await api.explanation(selected)); }; return <div className="page-stack"><SectionHeader title="Explainability" subtitle="Attribution is loaded from fitted artifacts, never hard-coded." /><div className="control-panel"><label>Experiment<select value={selected} onChange={(event) => setSelected(event.target.value)}><option value="">Select experiment</option>{experiments.map((item) => <option key={item.experiment_id} value={item.experiment_id}>{item.model_name} · {item.experiment_id.slice(0, 8)}</option>)}</select></label><button className="primary-button" onClick={() => void load()}>Load explanation</button></div><div className="panel"><SectionHeader title="Feature attribution" subtitle={result?.note ?? "Select a persisted experiment."} />{result?.feature_importance?.length ? <div className="table-wrap"><table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>{result.feature_importance.slice(0, 20).map((row: any) => <tr key={row.feature}><td>{row.feature}</td><td>{(row.importance ?? row.mean_abs_shap ?? 0).toFixed(5)}</td></tr>)}</tbody></table></div> : <EmptyState>No compatible saved explanation is available.</EmptyState>}</div></div>; }

function ScenarioPage({ experiments }: { experiments: Experiment[] }) { const [selected, setSelected] = useState(experiments[0]?.experiment_id ?? ""); const [occupancy, setOccupancy] = useState(20); const [temperature, setTemperature] = useState(0); const [result, setResult] = useState<ScenarioResult | null>(null); const [status, setStatus] = useState(""); const run = async () => { if (!selected) { setStatus("Select an experiment with a saved checkpoint."); return; } setStatus("Running scenario…"); try { setResult(await api.scenario({ experiment_id: selected, occupancy_multiplier: 1 + occupancy / 100, temperature_delta_c: temperature, humidity_delta_pct: 0 })); setStatus("Scenario completed."); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Scenario failed"); } }; return <div className="page-stack"><SectionHeader title="Scenario simulator" subtitle="Model-based what-if analysis, not causal energy-savings evidence." action={<span className="soft-badge orange">Non-causal</span>} /><div className="two-column-grid"><div className="panel"><SectionHeader title="Scenario inputs" /><label>Experiment<select value={selected} onChange={(event) => setSelected(event.target.value)}><option value="">Select experiment</option>{experiments.map((item) => <option key={item.experiment_id} value={item.experiment_id}>{item.model_name}</option>)}</select></label><label className="range-label"><span>Occupancy change <strong>{occupancy > 0 ? "+" : ""}{occupancy}%</strong></span><input type="range" min={-50} max={100} value={occupancy} onChange={(event) => setOccupancy(Number(event.target.value))} /></label><label className="range-label"><span>Temperature delta <strong>{temperature > 0 ? "+" : ""}{temperature}°C</strong></span><input type="range" min={-10} max={10} value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label><button className="primary-button" onClick={() => void run()}>Run model scenario →</button><div className="notice-line">{status}</div></div><div className="panel scenario-result"><span className="card-label">MODEL-BASED OUTPUT</span>{result ? <><strong>{result.difference[0]?.[0]?.toFixed(3) ?? "—"}</strong><p>{result.interpretation}</p></> : <><strong>—</strong><p>Predictions appear after selecting a fitted experiment with a saved checkpoint.</p></>}</div></div></div>; }

function ReportsPage({ experiments }: { experiments: Experiment[] }) { const [status, setStatus] = useState(""); const generate = async (id: string) => { try { const result = await api.report(id, "Campus Energy Forecasting Research Report"); setStatus(`Report generated: ${result.path}`); window.open(`/api/reports/${id}.html`, "_blank", "noopener,noreferrer"); } catch (reason) { setStatus(reason instanceof Error ? reason.message : "Report generation failed"); } }; return <div className="page-stack"><SectionHeader title="Research reports" subtitle="HTML reports contain configuration, actual metrics, limitations and reproducibility metadata." /><div className="notice-line">{status}</div><div className="panel"><div className="table-wrap"><table><thead><tr><th>Experiment</th><th>Model</th><th>Status</th><th>Report</th></tr></thead><tbody>{experiments.length ? experiments.map((experiment) => <tr key={experiment.experiment_id}><td className="mono">{experiment.experiment_id.slice(0, 12)}</td><td className="capitalize">{experiment.model_name.replaceAll("_", " ")}</td><td><span className="soft-badge teal">{experiment.status}</span></td><td><button className="link-button" onClick={() => void generate(experiment.experiment_id)}>Generate HTML</button></td></tr>) : <tr><td colSpan={4} className="empty-cell">No actual experiment records available.</td></tr>}</tbody></table></div></div></div>; }

export default App;
