import { Activity, Gauge, Leaf, Scale, TrendingUp, Zap } from "lucide-react";

import { SimulationMetrics } from "@/lib/types";

interface MetricsStripProps {
  metrics: SimulationMetrics | null;
}

function formatFixed(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "0.00";
}

export function MetricsStrip({ metrics }: MetricsStripProps): JSX.Element {
  const cards = [
    {
      label: "Cumulative Reward",
      value: formatFixed(metrics?.cumulative_reward ?? 0),
      hint: "Episode objective",
      icon: TrendingUp,
      accent: "text-cyan-300",
    },
    {
      label: "Average Price",
      value: formatFixed(metrics?.average_price ?? 0, 3),
      hint: "Currency / kWh",
      icon: Gauge,
      accent: "text-amber-300",
    },
    {
      label: "Peak Demand",
      value: formatFixed(metrics?.peak_demand ?? 0),
      hint: "System stress proxy",
      icon: Activity,
      accent: "text-rose-300",
    },
    {
      label: "Peak Supply",
      value: formatFixed(metrics?.peak_supply ?? 0),
      hint: "Distributed generation",
      icon: Zap,
      accent: "text-emerald-300",
    },
    {
      label: "Grid Import",
      value: formatFixed(metrics?.total_grid_import ?? 0),
      hint: "External dependency",
      icon: Scale,
      accent: "text-indigo-300",
    },
    {
      label: "Renewable Utilization",
      value: `${formatFixed((metrics?.average_renewable_utilization ?? 0) * 100, 1)}%`,
      hint: "Local clean energy share",
      icon: Leaf,
      accent: "text-lime-300",
    },
  ];

  return (
    <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
      {cards.map(({ label, value, hint, icon: Icon, accent }, index) => (
        <article
          key={label}
          className="panel-surface animate-reveal px-4 py-4"
          style={{ animationDelay: `${index * 40}ms` }}
        >
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs uppercase tracking-[0.18em] text-slate-300">{label}</p>
            <Icon className={`h-4 w-4 ${accent}`} />
          </div>
          <p className="font-display text-2xl leading-none text-white">{value}</p>
          <p className="mt-2 text-xs text-slate-400">{hint}</p>
        </article>
      ))}
    </section>
  );
}
