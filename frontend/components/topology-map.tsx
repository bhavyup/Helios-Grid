import clsx from "clsx";
import { useMemo } from "react";

import { SimulationObservation, TopologyNode, TopologyPayload } from "@/lib/types";

interface TopologyMapProps {
  topology?: TopologyPayload;
  observation?: SimulationObservation;
}

type Position = { x: number; y: number };

const DEFAULT_WIDTH = 1200;
const DEFAULT_HEIGHT = 760;
const DEFAULT_PADDING = 96;
const HOUSE_WIDTH = 74;
const HOUSE_HEIGHT = 54;
const SOLAR_WIDTH = 38;
const SOLAR_HEIGHT = 18;

function houseFillRatio(state: number[] | undefined, maxBattery: number): number {
  if (!state) {
    return 0;
  }
  const battery = Number.isFinite(state[3]) ? state[3] : 0;
  return Math.max(0, Math.min(1, battery / Math.max(maxBattery, 1e-6)));
}

function inferBounds(
  topology: TopologyPayload | undefined,
  positionedNodes: Array<TopologyNode & Position>,
): {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
} {
  if (topology?.bounds) {
    return {
      minX: topology.bounds.min_x,
      maxX: topology.bounds.max_x,
      minY: topology.bounds.min_y,
      maxY: topology.bounds.max_y,
      width: topology.bounds.width,
      height: topology.bounds.height,
    };
  }

  if (positionedNodes.length === 0) {
    return {
      minX: 0,
      maxX: DEFAULT_WIDTH,
      minY: 0,
      maxY: DEFAULT_HEIGHT,
      width: DEFAULT_WIDTH,
      height: DEFAULT_HEIGHT,
    };
  }

  const minX = Math.min(...positionedNodes.map((node) => node.x));
  const maxX = Math.max(...positionedNodes.map((node) => node.x));
  const minY = Math.min(...positionedNodes.map((node) => node.y));
  const maxY = Math.max(...positionedNodes.map((node) => node.y));
  const width = Math.max(maxX - minX + DEFAULT_PADDING * 2, DEFAULT_WIDTH * 0.8);
  const height = Math.max(maxY - minY + DEFAULT_PADDING * 2, DEFAULT_HEIGHT * 0.8);

  return {
    minX: minX - DEFAULT_PADDING,
    maxX: minX - DEFAULT_PADDING + width,
    minY: minY - DEFAULT_PADDING,
    maxY: minY - DEFAULT_PADDING + height,
    width,
    height,
  };
}

function buildFallbackPositions(topology: TopologyPayload | undefined, nodes: TopologyNode[]): Map<number, Position> {
  const bounds = topology?.bounds;
  const centerX = bounds ? bounds.min_x + bounds.width / 2 : DEFAULT_WIDTH / 2;
  const centerY = bounds ? bounds.min_y + bounds.height / 2 : DEFAULT_HEIGHT / 2;

  const positions = new Map<number, Position>();
  const households = nodes.filter((entry) => entry.type === "household");
  const solar = nodes.filter((entry) => entry.type === "solar");
  const wind = nodes.filter((entry) => entry.type === "wind");
  const houseLots = [
    { x: centerX - 240, y: centerY - 90 },
    { x: centerX + 120, y: centerY - 90 },
    { x: centerX - 240, y: centerY + 50 },
    { x: centerX + 120, y: centerY + 50 },
    { x: centerX - 100, y: centerY - 90 },
    { x: centerX + 260, y: centerY - 90 },
    { x: centerX - 100, y: centerY + 50 },
    { x: centerX + 260, y: centerY + 50 },
    { x: centerX - 240, y: centerY - 210 },
    { x: centerX + 120, y: centerY - 210 },
    { x: centerX - 240, y: centerY + 190 },
    { x: centerX + 120, y: centerY + 190 },
  ];
  const windSpots = [
    { x: centerX - 330, y: centerY - 210 },
    { x: centerX, y: centerY - 250 },
    { x: centerX + 330, y: centerY - 210 },
    { x: centerX - 330, y: centerY + 240 },
    { x: centerX + 330, y: centerY + 240 },
  ];

  const gridNode = nodes.find((entry) => entry.type === "grid");
  if (gridNode) {
    positions.set(gridNode.id, { x: centerX, y: centerY });
  }

  households.forEach((node, index) => {
    positions.set(node.id, houseLots[index % houseLots.length]);
  });

  solar.forEach((node, index) => {
    const anchor = households[index % Math.max(households.length, 1)];
    const anchorPosition = anchor ? positions.get(anchor.id) : undefined;
    positions.set(
      node.id,
      anchorPosition
        ? { x: anchorPosition.x, y: anchorPosition.y - 48 }
        : { x: centerX, y: centerY - 120 },
    );
  });

  wind.forEach((node, index) => {
    positions.set(node.id, windSpots[index % windSpots.length]);
  });

  return positions;
}

function renderHouseShape(position: Position, ratio: number): JSX.Element {
  const roofTop = position.y - HOUSE_HEIGHT / 2 - 14;
  const bodyTop = position.y - HOUSE_HEIGHT / 2;
  const bodyLeft = position.x - HOUSE_WIDTH / 2;
  const bodyRight = position.x + HOUSE_WIDTH / 2;
  const bodyBottom = position.y + HOUSE_HEIGHT / 2;

  return (
    <g>
      <rect
        x={bodyLeft - 8}
        y={bodyTop - 2}
        width={HOUSE_WIDTH + 16}
        height={HOUSE_HEIGHT + 16}
        rx={18}
        fill="rgba(255,255,255,0.05)"
      />
      <polygon
        points={`${bodyLeft},${bodyTop} ${position.x},${roofTop} ${bodyRight},${bodyTop}`}
        fill="#1f2937"
        opacity={0.92}
      />
      <rect
        x={bodyLeft}
        y={bodyTop}
        width={HOUSE_WIDTH}
        height={HOUSE_HEIGHT}
        rx={12}
        fill={`rgba(125, 211, 252, ${0.25 + ratio * 0.7})`}
        stroke="#dbeafe"
        strokeWidth={1.5}
      />
      <rect
        x={position.x - 17}
        y={bodyTop + 10}
        width={34}
        height={16}
        rx={4}
        fill="rgba(255,255,255,0.14)"
      />
      <rect
        x={position.x - 20}
        y={bodyBottom - 18}
        width={40}
        height={5}
        rx={2.5}
        fill={`rgba(15, 23, 42, ${0.4 + ratio * 0.5})`}
      />
    </g>
  );
}

function renderSolarShape(position: Position): JSX.Element {
  return (
    <g>
      <rect
        x={position.x - SOLAR_WIDTH / 2}
        y={position.y - SOLAR_HEIGHT / 2}
        width={SOLAR_WIDTH}
        height={SOLAR_HEIGHT}
        rx={4}
        fill="rgba(251, 191, 36, 0.94)"
        stroke="#fff7d6"
        strokeWidth={1.2}
      />
      <line
        x1={position.x - SOLAR_WIDTH / 2 + 5}
        y1={position.y - SOLAR_HEIGHT / 2 + 4}
        x2={position.x + SOLAR_WIDTH / 2 - 5}
        y2={position.y + SOLAR_HEIGHT / 2 - 4}
        stroke="rgba(255,255,255,0.24)"
        strokeWidth={1}
      />
      <line
        x1={position.x - SOLAR_WIDTH / 2 + 5}
        y1={position.y + SOLAR_HEIGHT / 2 - 4}
        x2={position.x + SOLAR_WIDTH / 2 - 5}
        y2={position.y - SOLAR_HEIGHT / 2 + 4}
        stroke="rgba(255,255,255,0.18)"
        strokeWidth={1}
      />
    </g>
  );
}

function renderWindShape(position: Position): JSX.Element {
  return (
    <g>
      <line
        x1={position.x}
        y1={position.y + 20}
        x2={position.x}
        y2={position.y - 22}
        stroke="#86efac"
        strokeWidth={4}
        strokeLinecap="round"
      />
      <circle cx={position.x} cy={position.y - 22} r={6} fill="#d1fae5" />
      <line
        x1={position.x}
        y1={position.y - 22}
        x2={position.x - 18}
        y2={position.y - 30}
        stroke="#d1fae5"
        strokeWidth={2}
        strokeLinecap="round"
      />
      <line
        x1={position.x}
        y1={position.y - 22}
        x2={position.x + 17}
        y2={position.y - 32}
        stroke="#d1fae5"
        strokeWidth={2}
        strokeLinecap="round"
      />
      <line
        x1={position.x}
        y1={position.y - 22}
        x2={position.x + 2}
        y2={position.y - 42}
        stroke="#d1fae5"
        strokeWidth={2}
        strokeLinecap="round"
      />
    </g>
  );
}

function renderGridHub(position: Position): JSX.Element {
  return (
    <g>
      <circle cx={position.x} cy={position.y} r={28} fill="rgba(45, 212, 191, 0.18)" />
      <circle cx={position.x} cy={position.y} r={18} fill="#2dd4bf" stroke="#ecfeff" strokeWidth={1.8} />
      <circle cx={position.x} cy={position.y} r={6} fill="#ecfeff" opacity={0.7} />
    </g>
  );
}

export function TopologyMap({ topology, observation }: TopologyMapProps): JSX.Element {
  const nodes = useMemo(() => topology?.nodes ?? [], [topology]);
  const edges = useMemo(() => topology?.edges ?? [], [topology]);
  const fallbackPositions = useMemo(() => buildFallbackPositions(topology, nodes), [nodes, topology]);

  const positionedNodes = useMemo(
    () =>
      nodes.map((node, index) => {
        const fallback = fallbackPositions.get(node.id) ?? { x: DEFAULT_WIDTH / 2, y: DEFAULT_HEIGHT / 2 };
        return {
          ...node,
          x: typeof node.x === "number" ? node.x : fallback.x,
          y: typeof node.y === "number" ? node.y : fallback.y,
          lot_index: typeof node.lot_index === "number" ? node.lot_index : index,
        };
      }),
    [fallbackPositions, nodes],
  );

  const gridNode = positionedNodes.find((node) => node.type === "grid");
  const households = positionedNodes.filter((node) => node.type === "household");
  const houseStates = observation?.house_states ?? [];
  const maxBattery =
    houseStates.length > 0 ? Math.max(...houseStates.map((state) => state[3] ?? 0), 1) : 1;

  const householdLookup = new Map<number, number[]>();
  households.forEach((household, index) => {
    householdLookup.set(household.id, houseStates[index]);
  });

  const bounds = inferBounds(topology, positionedNodes);
  const padding = 72;
  const viewBox = `${bounds.minX - padding} ${bounds.minY - padding} ${bounds.width + padding * 2} ${bounds.height + padding * 2}`;
  const hubPosition = gridNode ?? {
    x: bounds.minX + bounds.width / 2,
    y: bounds.minY + bounds.height / 2,
  };
  const verticalRoadX = hubPosition.x;
  const horizontalRoadY = hubPosition.y;

  return (
    <section className="panel-surface flex h-full flex-col px-5 py-5 lg:px-6 lg:py-6">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Neighborhood map</p>
          <h2 className="hero-display mt-2 text-3xl font-semibold tracking-[-0.03em] text-white">
            2-D Neighborhood View
          </h2>
          <p className="section-copy mt-2 text-sm">
            Building markers, rooftop assets, & service links on a local map grid.
          </p>
        </div>
        <div className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-slate-300">
          {topology?.node_count ?? 0} nodes / {topology?.edge_count ?? 0} edges
        </div>
      </div>

      <div className="relative flex-1 overflow-hidden rounded-[1.35rem] border border-white/10 bg-[rgba(255,255,255,0.02)]">
        <svg viewBox={viewBox} className="h-full w-full">
          <defs>
            <linearGradient id="mapBackground" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(15, 23, 42, 0.98)" />
              <stop offset="60%" stopColor="rgba(8, 15, 30, 0.98)" />
              <stop offset="100%" stopColor="rgba(2, 6, 23, 0.98)" />
            </linearGradient>
            <linearGradient id="roadGradient" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(226, 232, 240, 0.16)" />
              <stop offset="100%" stopColor="rgba(148, 163, 184, 0.07)" />
            </linearGradient>
            <pattern id="plotGrid" width="64" height="64" patternUnits="userSpaceOnUse">
              <path d="M 64 0 L 0 0 0 64" fill="none" stroke="rgba(148, 163, 184, 0.08)" strokeWidth="1" />
            </pattern>
            <filter id="softGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="10" result="blur" />
              <feColorMatrix
                in="blur"
                type="matrix"
                values="1 0 0 0 0.06 0 1 0 0 0.13 0 0 1 0 0.14 0 0 0 0.34 0"
              />
            </filter>
          </defs>

          <rect
            x={bounds.minX - padding}
            y={bounds.minY - padding}
            width={bounds.width + padding * 2}
            height={bounds.height + padding * 2}
            fill="url(#mapBackground)"
          />
          <rect
            x={bounds.minX - padding}
            y={bounds.minY - padding}
            width={bounds.width + padding * 2}
            height={bounds.height + padding * 2}
            fill="url(#plotGrid)"
          />

          <rect
            x={verticalRoadX - 30}
            y={bounds.minY - padding}
            width={60}
            height={bounds.height + padding * 2}
            fill="url(#roadGradient)"
            opacity={0.82}
          />
          <rect
            x={bounds.minX - padding}
            y={horizontalRoadY - 30}
            width={bounds.width + padding * 2}
            height={60}
            fill="url(#roadGradient)"
            opacity={0.82}
          />

          <rect
            x={bounds.minX + 42}
            y={bounds.minY + 34}
            width={bounds.width - 84}
            height={bounds.height - 68}
            rx={36}
            fill="rgba(255,255,255,0.03)"
            stroke="rgba(255,255,255,0.05)"
          />

          {edges.map((edge) => {
            const from = positionedNodes.find((node) => node.id === edge.source);
            const to = positionedNodes.find((node) => node.id === edge.target);
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
                stroke="rgba(125, 211, 252, 0.28)"
                strokeDasharray="6 8"
                strokeLinecap="round"
                strokeWidth={1.8}
              />
            );
          })}

          {positionedNodes.map((node) => {
            const isGrid = node.type === "grid";
            const ratio = houseFillRatio(householdLookup.get(node.id), maxBattery);
            const labelOffsetY = isGrid ? 38 : node.type === "solar" ? 26 : 34;

            return (
              <g key={node.id} filter={isGrid ? "url(#softGlow)" : undefined}>
                {isGrid ? renderGridHub(node) : null}
                {!isGrid && node.type === "household" ? renderHouseShape(node, ratio) : null}
                {!isGrid && node.type === "solar" ? renderSolarShape(node) : null}
                {!isGrid && node.type === "wind" ? renderWindShape(node) : null}
                {!isGrid && node.type !== "household" && node.type !== "solar" && node.type !== "wind" ? (
                  <circle cx={node.x} cy={node.y} r={10} fill="#86efac" stroke="#d1fae5" strokeWidth={1.4} />
                ) : null}
                <text
                  x={node.x}
                  y={node.y + labelOffsetY}
                  textAnchor="middle"
                  className="fill-slate-300 text-[10px] tracking-[0.14em]"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-[9px] uppercase tracking-[0.15em] text-slate-300">
        <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Grid hub: teal</span>
        <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Household buildings: cyan intensity</span>
        <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Solar roofs: amber</span>
        <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-2">Edge assets: green</span>
      </div>

      <p className={clsx("mt-3 text-xs", nodes.length === 0 ? "text-rose-300" : "text-slate-400")}>
        {nodes.length === 0
          ? "Topology is unavailable. Reset simulation once to hydrate graph state."
          : topology?.layout === "neighborhood-grid"
            ? "Local coordinates are rendered from the topology payload and can later be swapped for GIS-backed coordinates."
            : "Node saturation is estimated from current household battery state at each timestep."}
      </p>
    </section>
  );
}
