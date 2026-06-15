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
    <section className="panel-surface overflow-visible z-40 h-full px-5 py-5 lg:px-6 lg:py-6">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-3xl">
          <p className="section-eyebrow">Grid timeline</p>
          <h2 className="hero-display mt-2 text-3xl font-semibold tracking-[-0.03em] text-white">Energy flow without visual noise</h2>
          <p className="section-copy mt-2 text-sm">Supply-demand balance, price evolution, reward trend, and renewable utilization are split into two calm charts.</p>
        </div>
        <div className="flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-300">
          <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Supply / demand</span>
          <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Price / reward</span>
          <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Renewables</span>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="panel-frame rounded-[1.4rem] p-3 lg:p-4">
          <h3 className="section-eyebrow mb-2 text-[10px]">Flow balance</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={rows}>
                <defs>
                  <linearGradient id="supplyGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#7fb6a8" stopOpacity={0.78} />
                    <stop offset="95%" stopColor="#7fb6a8" stopOpacity={0.08} />
                  </linearGradient>
                  <linearGradient id="demandGradient" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="5%" stopColor="#f0c46b" stopOpacity={0.72} />
                    <stop offset="95%" stopColor="#f0c46b" stopOpacity={0.08} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="label" stroke="#a8b0bc" tick={{ fontSize: 11 }} />
                <YAxis stroke="#a8b0bc" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{
                    background: "rgba(9, 12, 17, 0.97)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: 16,
                  }}
                />
                <Legend wrapperStyle={{ color: "#dbe2ea" }} />
                <Area
                  type="monotone"
                  dataKey="supply"
                  name="Supply"
                  stroke="#7fb6a8"
                  fill="url(#supplyGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="demand"
                  name="Demand"
                  stroke="#f0c46b"
                  fill="url(#demandGradient)"
                  strokeWidth={2}
                />
                <Area
                  type="monotone"
                  dataKey="grid_import"
                  name="Grid Import"
                  stroke="#89a5d1"
                  fill="rgba(137, 165, 209, 0.12)"
                  strokeWidth={1.8}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel-frame rounded-[1.4rem] p-3 lg:p-4">
          <h3 className="section-eyebrow mb-2 text-[10px]">Price / reward / renewable share</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rows}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="label" stroke="#a8b0bc" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" stroke="#a8b0bc" tick={{ fontSize: 11 }} />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#a8b0bc"
                  tick={{ fontSize: 11 }}
                  domain={[0, 1]}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgba(9, 12, 17, 0.97)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: 16,
                  }}
                />
                <Legend wrapperStyle={{ color: "#dbe2ea" }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="price"
                  name="Price"
                  stroke="#89a5d1"
                  strokeWidth={2.1}
                  dot={false}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="reward"
                  name="Reward"
                  stroke="#f0c46b"
                  strokeWidth={2.1}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="renewable_utilization"
                  name="Renewable Utilization"
                  stroke="#7fb6a8"
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
