import clsx from "clsx";

import { SimulationObservation, TopologyPayload } from "@/lib/types";

interface TopologyMapProps {
  topology?: TopologyPayload;
  observation?: SimulationObservation;
}

type Position = { x: number; y: number };

const WIDTH = 760;
const HEIGHT = 460;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;

function houseFillRatio(state: number[] | undefined, maxBattery: number): number {
  if (!state) {
    return 0;
  }
  const battery = Number.isFinite(state[3]) ? state[3] : 0;
  return Math.max(0, Math.min(1, battery / Math.max(maxBattery, 1e-6)));
}

export function TopologyMap({ topology, observation }: TopologyMapProps): JSX.Element {
  const nodes = topology?.nodes ?? [];
  const edges = topology?.edges ?? [];

  const gridNode = nodes.find((node) => node.type === "grid");
  const households = nodes.filter((node) => node.type === "household");
  const producers = nodes.filter((node) => node.type !== "grid" && node.type !== "household");

  const houseStates = observation?.house_states ?? [];
  const maxBattery = houseStates.length > 0 ? Math.max(...houseStates.map((state) => state[3] ?? 0), 1) : 1;

  const positions = new Map<number, Position>();

  if (gridNode) {
    positions.set(gridNode.id, { x: CENTER_X, y: CENTER_Y });
  }

  households.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(households.length, 1);
    const radius = 160;
    positions.set(node.id, {
      x: CENTER_X + Math.cos(angle) * radius,
      y: CENTER_Y + Math.sin(angle) * radius,
    });
  });

  producers.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / Math.max(producers.length, 1);
    const radius = 245;
    positions.set(node.id, {
      x: CENTER_X + Math.cos(angle) * radius,
      y: CENTER_Y + Math.sin(angle) * radius,
    });
  });

  const householdLookup = new Map<number, number[]>();
  households.forEach((household, index) => {
    householdLookup.set(household.id, houseStates[index]);
  });

  return (
    <section className="panel-surface h-full px-5 py-5">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="panel-title">Neighborhood Topology</h2>
          <p className="mt-1 text-xs text-slate-400">Battery saturation and transfer corridors in the active grid graph.</p>
        </div>
        <div className="rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-slate-300">
          {topology?.node_count ?? 0} nodes / {topology?.edge_count ?? 0} edges
        </div>
      </div>

      <div className="relative overflow-hidden rounded-xl border border-slate-700/70 bg-slate-950/70">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="h-[360px] w-full">
          <defs>
            <linearGradient id="edgeStroke" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(39, 205, 255, 0.15)" />
              <stop offset="100%" stopColor="rgba(255, 191, 73, 0.25)" />
            </linearGradient>
          </defs>

          {edges.map((edge) => {
            const from = positions.get(edge.source);
            const to = positions.get(edge.target);
            if (!from || !to) {
              return null;
            }
            return (
              <line
                key={`${edge.source}-${edge.target}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke="url(#edgeStroke)"
                strokeWidth={1.8}
              />
            );
          })}

          {nodes.map((node) => {
            const position = positions.get(node.id);
            if (!position) {
              return null;
            }

            const isGrid = node.type === "grid";
            const isHouse = node.type === "household";
            const ratio = houseFillRatio(householdLookup.get(node.id), maxBattery);

            const fill = isGrid
              ? "#2dd4bf"
              : isHouse
                ? `rgba(56, 189, 248, ${0.25 + ratio * 0.75})`
                : node.type === "solar"
                  ? "#fbbf24"
                  : "#86efac";

            const stroke = isGrid ? "#99f6e4" : "#cbd5e1";
            const radius = isGrid ? 13 : 9;

            return (
              <g key={node.id}>
                <circle cx={position.x} cy={position.y} r={radius + 9} fill="rgba(148, 163, 184, 0.08)" />
                <circle cx={position.x} cy={position.y} r={radius} fill={fill} stroke={stroke} strokeWidth={1.4} />
                <text
                  x={position.x}
                  y={position.y + 26}
                  textAnchor="middle"
                  className="fill-slate-300 text-[10px] tracking-[0.1em]"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.15em] text-slate-300">
        <span className="rounded-sm border border-slate-700 bg-slate-900/60 px-2 py-1">Grid Hub: teal</span>
        <span className="rounded-sm border border-slate-700 bg-slate-900/60 px-2 py-1">Household Battery: cyan intensity</span>
        <span className="rounded-sm border border-slate-700 bg-slate-900/60 px-2 py-1">Solar: amber</span>
        <span className="rounded-sm border border-slate-700 bg-slate-900/60 px-2 py-1">Wind/Other: green</span>
      </div>

      <p
        className={clsx(
          "mt-3 text-xs",
          nodes.length === 0 ? "text-rose-300" : "text-slate-400",
        )}
      >
        {nodes.length === 0
          ? "Topology is unavailable. Reset simulation once to hydrate graph state."
          : "Node saturation is estimated from current household battery state at each timestep."}
      </p>
    </section>
  );
}
