import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { TrajectoryPoint } from "@/lib/types";

interface EnergyChartsProps {
  history: TrajectoryPoint[];
}

function chartRows(history: TrajectoryPoint[]): Array<TrajectoryPoint & { label: string }> {
  return history.map((point) => ({
    ...point,
    label: `t${point.step}`,
  }));
}

export function EnergyCharts({ history }: EnergyChartsProps): JSX.Element {
  const rows = chartRows(history);

  return (
    <section className="panel-surface h-full px-5 py-5">
      <div className="mb-3">
        <h2 className="panel-title">Grid Dynamics Timeline</h2>
        <p className="mt-1 text-xs text-slate-400">Supply-demand balance, price evolution, reward trend and renewable utilization.</p>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-xl border border-slate-700/70 bg-slate-950/60 p-3">
          <h3 className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-300">Flow Balance</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rows}>
                <defs>
                  <linearGradient id="supplyGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0.08} />
                  </linearGradient>
                  <linearGradient id="demandGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.7} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0.08} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                <XAxis dataKey="label" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 23, 42, 0.95)",
                    border: "1px solid rgba(100, 116, 139, 0.5)",
                    borderRadius: 8,
                  }}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="supply"
                  name="Supply"
                  stroke="#2dd4bf"
                  fill="url(#supplyGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="demand"
                  name="Demand"
                  stroke="#f97316"
                  fill="url(#demandGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="grid_import"
                  name="Grid Import"
                  stroke="#a78bfa"
                  fill="rgba(167, 139, 250, 0.12)"
                  strokeWidth={1.8}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-700/70 bg-slate-950/60 p-3">
          <h3 className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-300">Price, Reward, Renewable Share</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                <XAxis dataKey="label" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#94a3b8"
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(15, 23, 42, 0.95)",
                    border: "1px solid rgba(100, 116, 139, 0.5)",
                    borderRadius: 8,
                  }}
                />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="price"
                  name="Price"
                  stroke="#38bdf8"
                  strokeWidth={2.1}
                  dot={false}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="reward"
                  name="Reward"
                  stroke="#f59e0b"
                  strokeWidth={2.1}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="renewable_utilization"
                  name="Renewable Utilization"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
