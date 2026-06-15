"use client";

import { useThree } from "@react-three/fiber";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  OrbitControls,
  Html,
  QuadraticBezierLine,
  useCursor,
} from "@react-three/drei";
import { useMemo, useRef, useState, useCallback, useEffect } from "react";
import { useSceneDirectorStore } from "@/store/useSceneDirectorStore";

import {
  SimulationObservation,
  TopologyNode,
  TopologyPayload,
} from "@/lib/types";

interface Neighborhood3DSceneProps {
  topology?: TopologyPayload;
  observation?: SimulationObservation;
  latestInfo?: Record<string, unknown>;
  step?: number;
}

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
}

interface WorldNode extends TopologyNode {
  position: [number, number, number];
  orderIndex: number;
}

interface HouseStateRaw {
  energy: number;
  consumption: number;
  production: number;
  batteryLevel: number;
  price: number;
  gridImport: number;
  p2pBuy: number;
  p2pSell: number;
  timeNorm: number;
  netBalance: number;
}

interface HouseStateSummary {
  production: number;
  battery: number;
  gridImport: number;
  p2pBuy: number;
  p2pSell: number;
}

interface FlowLink {
  source: [number, number, number];
  target: [number, number, number];
  color: string;
  pulseSpeed: number;
  pulsePhase: number;

  kind: "grid" | "order" | "trade";
  quantity?: number;
  price?: number;
  buyerId?: number;
  sellerId?: number;
  householdId?: number; // for "grid" and "order" links
}

interface DayCycle {
  daylight: number;
  progress: number;
  source: "weather" | "step";
}

interface P2POrder {
  household_id: number;
  side: "buy" | "sell";
  quantity: number;
  limit_price: number;
}

interface P2PTrade {
  buyer_household_id: number;
  seller_household_id: number;
  quantity: number;
  price: number;
}

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function asFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function getMarketSnapshotObject(
  latestInfo?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const candidate = latestInfo?.market_snapshot;
  if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
    return candidate as Record<string, unknown>;
  }
  return undefined;
}

function parseP2PTrades(marketSnapshot?: Record<string, unknown>): P2PTrade[] {
  const raw = marketSnapshot?.["p2p_trades"];
  if (!Array.isArray(raw)) return [];

  const trades: P2PTrade[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const obj = item as Record<string, unknown>;

    const buyer = asFiniteNumber(obj["buyer_household_id"]);
    const seller = asFiniteNumber(obj["seller_household_id"]);
    const quantity = asFiniteNumber(obj["quantity"]);
    const price = asFiniteNumber(obj["price"]);

    if (
      buyer === undefined ||
      seller === undefined ||
      quantity === undefined ||
      price === undefined
    ) {
      continue;
    }

    trades.push({
      buyer_household_id: buyer,
      seller_household_id: seller,
      quantity,
      price,
    });
  }

  return trades;
}

function parseP2POrders(marketSnapshot?: Record<string, unknown>): P2POrder[] {
  const raw = marketSnapshot?.["p2p_orders"];
  if (!Array.isArray(raw)) return [];

  const orders: P2POrder[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object" || Array.isArray(item)) continue;
    const obj = item as Record<string, unknown>;

    const householdId = asFiniteNumber(obj["household_id"]);
    const sideRaw = obj["side"];
    const quantity = asFiniteNumber(obj["quantity"]);
    const limitPrice = asFiniteNumber(obj["limit_price"]);

    const side = sideRaw === "buy" || sideRaw === "sell" ? sideRaw : undefined;

    if (
      householdId === undefined ||
      side === undefined ||
      quantity === undefined ||
      limitPrice === undefined
    ) {
      continue;
    }

    orders.push({
      household_id: householdId,
      side,
      quantity,
      limit_price: limitPrice,
    });
  }

  return orders;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

type RoadPlan = {
  verticalXs: number[];
  horizontalZs: number[];
  spanX: number;
  spanZ: number;
};

function deriveRoadPlanFromGrid(houses: WorldNode[]): RoadPlan {
  // group by col/row if present
  const byCol = new Map<number, number[]>();
  const byRow = new Map<number, number[]>();

  for (const h of houses) {
    if (typeof h.col === "number") {
      const arr = byCol.get(h.col) ?? [];
      arr.push(h.position[0]);
      byCol.set(h.col, arr);
    }
    if (typeof h.row === "number") {
      const arr = byRow.get(h.row) ?? [];
      arr.push(h.position[2]);
      byRow.set(h.row, arr);
    }
  }

  // fallback: if row/col missing for some reason, use unique rounded coordinates
  const cols = [...byCol.keys()].sort((a, b) => a - b);
  const rows = [...byRow.keys()].sort((a, b) => a - b);

  const colXs = cols.length
    ? cols.map((c) => average(byCol.get(c) ?? []))
    : [...new Set(houses.map((h) => Math.round(h.position[0] * 2) / 2))].sort(
        (a, b) => a - b,
      );

  const rowZs = rows.length
    ? rows.map((r) => average(byRow.get(r) ?? []))
    : [...new Set(houses.map((h) => Math.round(h.position[2] * 2) / 2))].sort(
        (a, b) => a - b,
      );

  const colDiffs = colXs.slice(1).map((x, i) => x - colXs[i]);
  const rowDiffs = rowZs.slice(1).map((z, i) => z - rowZs[i]);
  const colSpacing = median(colDiffs) || 4.5;
  const rowSpacing = median(rowDiffs) || 3.8;

  const verticalXs: number[] = [];
  const horizontalZs: number[] = [];

  // perimeter roads (outside the first/last columns/rows)
  if (colXs.length) {
    verticalXs.push(colXs[0] - colSpacing / 2);
    for (let i = 0; i < colXs.length - 1; i += 1) {
      verticalXs.push((colXs[i] + colXs[i + 1]) / 2);
    }
    verticalXs.push(colXs[colXs.length - 1] + colSpacing / 2);
  }

  if (rowZs.length) {
    horizontalZs.push(rowZs[0] - rowSpacing / 2);
    for (let i = 0; i < rowZs.length - 1; i += 1) {
      horizontalZs.push((rowZs[i] + rowZs[i + 1]) / 2);
    }
    horizontalZs.push(rowZs[rowZs.length - 1] + rowSpacing / 2);
  }

  const spanX = colXs.length
    ? colXs[colXs.length - 1] - colXs[0] + colSpacing
    : WORLD_WIDTH;
  const spanZ = rowZs.length
    ? rowZs[rowZs.length - 1] - rowZs[0] + rowSpacing
    : WORLD_DEPTH;

  return { verticalXs, horizontalZs, spanX, spanZ };
}

const TAU = Math.PI * 2;
const DEFAULT_WIDTH = 1000;
const DEFAULT_HEIGHT = 700;
const WORLD_WIDTH = 50;
const WORLD_DEPTH = 36;
const ROAD_THICKNESS = 0.08;
const ROAD_WIDTH = 0.95;
const SIDEWALK_THICKNESS = 0.06;
const SAMPLE_DAY_STEPS = 96;
const NEAREST_SOLAR_RADIUS = 7.5;

const HOUSE_COLORS = [
  "#d7c2ad",
  "#dcd9d1",
  "#c1ced8",
  "#d4b396",
  "#c9c5df",
  "#b8cdbd",
];

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function mix(a: number, b: number, amount: number): number {
  return a + (b - a) * amount;
}

function getBounds(topology?: TopologyPayload): Bounds {
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

  return {
    minX: 0,
    maxX: DEFAULT_WIDTH,
    minY: 0,
    maxY: DEFAULT_HEIGHT,
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
  };
}

function toWorldPosition(
  node: TopologyNode,
  bounds: Bounds,
): [number, number, number] {
  const centerX = bounds.minX + bounds.width / 2;
  const centerY = bounds.minY + bounds.height / 2;
  const normalizedX =
    typeof node.x === "number"
      ? (node.x - centerX) / Math.max(bounds.width, 1)
      : 0;
  const normalizedY =
    typeof node.y === "number"
      ? (node.y - centerY) / Math.max(bounds.height, 1)
      : 0;

  return [normalizedX * WORLD_WIDTH, 0, normalizedY * WORLD_DEPTH];
}

function houseColor(index: number): string {
  return HOUSE_COLORS[index % HOUSE_COLORS.length];
}

function getWeatherObject(
  latestInfo?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const weather = latestInfo?.weather;
  if (weather && typeof weather === "object" && !Array.isArray(weather)) {
    return weather as Record<string, unknown>;
  }

  return undefined;
}

function getLatestNumeric(
  latestInfo: Record<string, unknown> | undefined,
  keys: string[],
): number | undefined {
  if (!latestInfo) {
    return undefined;
  }

  for (const key of keys) {
    const value = latestInfo[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  const weather = getWeatherObject(latestInfo);
  if (!weather) {
    return undefined;
  }

  for (const key of keys) {
    const value = weather[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return undefined;
}

function resolveDayCycle(
  latestInfo?: Record<string, unknown>,
  step?: number,
): DayCycle {
  const weather = getWeatherObject(latestInfo);
  const timestampCandidates = [
    weather?.utc_timestamp,
    weather?.timestamp,
    latestInfo?.weather_timestamp,
    latestInfo?.utc_timestamp,
    latestInfo?.timestamp,
  ];

  for (const candidate of timestampCandidates) {
    if (candidate === undefined || candidate === null) {
      continue;
    }

    const parsed = new Date(String(candidate));
    if (Number.isNaN(parsed.getTime())) {
      continue;
    }

    const baseSeconds =
      parsed.getUTCHours() * 3600 +
      parsed.getUTCMinutes() * 60 +
      parsed.getUTCSeconds();
    const stepIndex = step ?? 0;
    const stepSeconds = stepIndex * 15 * 60;

    const totalSeconds = (baseSeconds + stepSeconds) % 86400;
    const progress = totalSeconds / 86400;

    // shift so ~6am is sunrise, noon is brightest
    const shifted = progress - 0.25;
    const daylight = clamp(Math.sin(shifted * TAU) * 0.5 + 0.5, 0, 1);
    return {
      daylight,
      progress,
      source: "weather",
    };
  }

  const stepIndex =
    step ??
    getLatestNumeric(latestInfo, [
      "step",
      "current_time",
      "weather_index_used",
    ]) ??
    0;
  const progress =
    (((stepIndex % SAMPLE_DAY_STEPS) + SAMPLE_DAY_STEPS) % SAMPLE_DAY_STEPS) /
    SAMPLE_DAY_STEPS;
  return {
    daylight: clamp(Math.sin(progress * TAU - Math.PI / 2) * 0.5 + 0.5, 0, 1),
    progress,
    source: "step",
  };
}

function normalizeSignal(value: unknown): number {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }

  const magnitude = Math.abs(value);
  if (magnitude <= 1) {
    return clamp(magnitude, 0, 1);
  }

  return clamp(magnitude / (magnitude + 8), 0, 1);
}

function buildHouseRawStateMap(
  nodes: WorldNode[],
  observation?: SimulationObservation,
): Map<number, HouseStateRaw> {
  const houseStates = observation?.house_states ?? [];
  const stateMap = new Map<number, HouseStateRaw>();

  nodes
    .filter((node) => node.type === "household")
    .forEach((node, index) => {
      const lotIndex =
        typeof node.lot_index === "number" ? node.lot_index : index;
      const state = houseStates[lotIndex] ?? houseStates[index];
      if (!state) return;

      stateMap.set(node.id, {
        energy: Number(state[0] ?? 0),
        consumption: Number(state[1] ?? 0),
        production: Number(state[2] ?? 0),
        batteryLevel: Number(state[3] ?? 0),
        price: Number(state[4] ?? 0),
        gridImport: Number(state[5] ?? 0),
        p2pBuy: Number(state[6] ?? 0),
        p2pSell: Number(state[7] ?? 0),
        timeNorm: Number(state[8] ?? 0),
        netBalance: Number(state[9] ?? 0),
      });
    });

  return stateMap;
}

function buildHouseStateMap(
  nodes: WorldNode[],
  observation?: SimulationObservation,
): Map<number, HouseStateSummary> {
  const houseStates = observation?.house_states ?? [];
  const stateMap = new Map<number, HouseStateSummary>();

  nodes
    .filter((node) => node.type === "household")
    .forEach((node, index) => {
      const lotIndex =
        typeof node.lot_index === "number" ? node.lot_index : index;
      const state = houseStates[lotIndex] ?? houseStates[index];
      if (!state) {
        return;
      }

      stateMap.set(node.id, {
        production: normalizeSignal(state[2]),
        battery: normalizeSignal(state[3]),
        gridImport: normalizeSignal(state[5]),
        p2pBuy: normalizeSignal(state[6]),
        p2pSell: normalizeSignal(state[7]),
      });
    });

  return stateMap;
}

function distanceXZ(
  left: [number, number, number],
  right: [number, number, number],
): number {
  return Math.hypot(left[0] - right[0], left[2] - right[2]);
}

function findNearestSolarNode(
  house: WorldNode,
  solarNodes: WorldNode[],
): { node: WorldNode; distance: number } | undefined {
  let closest: { node: WorldNode; distance: number } | undefined;

  for (const solarNode of solarNodes) {
    const distance = distanceXZ(house.position, solarNode.position);
    if (!closest || distance < closest.distance) {
      closest = { node: solarNode, distance };
    }
  }

  if (!closest || closest.distance > NEAREST_SOLAR_RADIUS) {
    return undefined;
  }

  return closest;
}

function buildFlowLinks(
  houses: WorldNode[],
  states: Map<number, HouseStateSummary>,
  hubPosition: [number, number, number],
  marketSnapshot?: Record<string, unknown>,
): FlowLink[] {
  const links: FlowLink[] = [];

  // 1) Grid import flows (keep your existing behavior)
  const importers = houses
    .map((house) => ({ house, state: states.get(house.id) }))
    .filter(({ state }) => (state?.gridImport ?? 0) > 0.08)
    .sort(
      (left, right) =>
        (right.state?.gridImport ?? 0) - (left.state?.gridImport ?? 0),
    );

  importers.slice(0, 12).forEach((entry, index) => {
    links.push({
      source: entry.house.position,
      target: hubPosition,
      color: index % 2 === 0 ? "#38bdf8" : "#60a5fa",
      pulseSpeed: 0.12 + index * 0.01,
      pulsePhase: index * 0.09,
      kind: "grid",
      householdId: entry.house.id,
    });
  });

  // Build lookup for real trade endpoints
  const houseById = new Map<number, WorldNode>();
  for (const h of houses) houseById.set(h.id, h);

  // 2) Real matched trades (append, do not return)
  const trades = parseP2PTrades(marketSnapshot);
  const cda =
    asFiniteNumber(marketSnapshot?.["p2p_cda_price"]) ??
    asFiniteNumber(marketSnapshot?.["clearing_price"]) ??
    0;

  if (trades.length) {
    const prices = trades.map((t) => t.price);
    const minP = Math.min(...prices);
    const maxP = Math.max(...prices);

    [...trades]
      .sort((a, b) => b.quantity - a.quantity)
      .slice(0, 40) // allow more; HUD will clip later
      .forEach((trade, index) => {
        const seller = houseById.get(trade.seller_household_id);
        const buyer = houseById.get(trade.buyer_household_id);
        if (!seller || !buyer) return;

        const qtyFactor = clamp(trade.quantity, 0, 1);
        const color = tradeColorFromPrice({
          price: trade.price,
          cda,
          min: minP,
          max: maxP,
        });

        links.push({
          source: seller.position,
          target: buyer.position,
          color,
          pulseSpeed: 0.08 + qtyFactor * 0.12 + index * 0.003,
          pulsePhase: index * 0.11,
          kind: "trade",
          quantity: trade.quantity,
          price: trade.price,
          buyerId: trade.buyer_household_id,
          sellerId: trade.seller_household_id,
        });
      });
  }

  //   return links;
  // }

  // 3) Orders (append, do not return)
  const orders = parseP2POrders(marketSnapshot);
  if (orders.length) {
    [...orders]
      .sort((a, b) => b.quantity - a.quantity)
      .slice(0, 40)
      .forEach((order, index) => {
        const house = houseById.get(order.household_id);
        if (!house) return;

        links.push({
          source: house.position,
          target: hubPosition,
          color: order.side === "buy" ? "#60a5fa" : "#2dd4bf",
          pulseSpeed: 0.09 + clamp(order.quantity, 0, 1) * 0.12 + index * 0.002,
          pulsePhase: index * 0.13,
          kind: "order",
          householdId: order.household_id,
          quantity: order.quantity,
          price: order.limit_price,
        });
      });
  }

  //   return links;
  // }

  // 4) Otherwise fall back to your existing synthetic pairing using state p2p signals
  if (!trades.length && !orders.length) {
    const sellers = houses
      .map((house) => ({ house, state: states.get(house.id) }))
      .filter(({ state }) => (state?.p2pSell ?? 0) > 0.08)
      .sort(
        (left, right) =>
          (right.state?.p2pSell ?? 0) - (left.state?.p2pSell ?? 0),
      );

    const buyers = houses
      .map((house) => ({ house, state: states.get(house.id) }))
      .filter(({ state }) => (state?.p2pBuy ?? 0) > 0.08)
      .sort(
        (left, right) => (right.state?.p2pBuy ?? 0) - (left.state?.p2pBuy ?? 0),
      );

    const matchCount = Math.min(12, sellers.length, buyers.length);
    for (let index = 0; index < matchCount; index += 1) {
      const seller = sellers[index].house;
      const buyer = buyers[(index * 3) % buyers.length].house;
      links.push({
        source: seller.position,
        target: buyer.position,
        color: index % 2 === 0 ? "#2dd4bf" : "#fbbf24",
        pulseSpeed: 0.09 + index * 0.008,
        pulsePhase: index * 0.11,
        kind: "trade",
        sellerId: seller.id,
        buyerId: buyer.id,
      });
    }
  }

  return links;
}

function Ground(): JSX.Element {
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
      position={[0, -0.05, 0]}
    >
      <planeGeometry args={[84, 58]} />
      <meshStandardMaterial color="#08111f" roughness={1} metalness={0} />
    </mesh>
  );
}

function Road({
  position,
  size,
}: {
  position: [number, number, number];
  size: [number, number, number];
}): JSX.Element {
  return (
    <mesh position={position} receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial color="#2f3747" roughness={0.96} metalness={0.02} />
    </mesh>
  );
}

function Sidewalk({
  position,
  size,
}: {
  position: [number, number, number];
  size: [number, number, number];
}): JSX.Element {
  return (
    <mesh position={position} receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial
        color="#8b98aa"
        roughness={0.95}
        metalness={0.02}
      />{" "}
    </mesh>
  );
}

function RoadMarkings({
  horizontal = true,
}: {
  horizontal?: boolean;
}): JSX.Element {
  return (
    <group>
      {Array.from({ length: 11 }, (_, index) => index - 5).map(
        (mark, index) => (
          <mesh
            key={`${horizontal ? "h" : "v"}-${mark}`}
            position={
              horizontal ? [mark * 3.15, 0.09, 0] : [0, 0.09, mark * 2.65]
            }
            rotation={[-Math.PI / 2, 0, 0]}
            receiveShadow
          >
            <planeGeometry args={[0.42, 1.45]} />
            <meshStandardMaterial
              color={index % 2 === 0 ? "#f8fafc" : "#e2e8f0"}
              emissive="#f8fafc"
              emissiveIntensity={0.08}
              transparent
              opacity={0.82}
              roughness={1}
              metalness={0}
            />
          </mesh>
        ),
      )}
    </group>
  );
}

function Fence({
  position,
}: {
  position: [number, number, number];
}): JSX.Element {
  return (
    <mesh position={position} castShadow>
      <boxGeometry args={[0.08, 0.42, 1.18]} />
      <meshStandardMaterial color="#d1d5db" roughness={0.84} metalness={0.08} />
    </mesh>
  );
}

function Tree({
  position,
}: {
  position: [number, number, number];
}): JSX.Element {
  return (
    <group position={position}>
      <mesh position={[0, 0.38, 0]} castShadow>
        <cylinderGeometry args={[0.06, 0.08, 0.74, 10]} />
        <meshStandardMaterial color="#7c4a2d" roughness={1} metalness={0} />
      </mesh>
      <mesh position={[0, 0.98, 0]} castShadow>
        <sphereGeometry args={[0.34, 18, 18]} />
        <meshStandardMaterial color="#22c55e" roughness={1} metalness={0} />
      </mesh>
      <mesh position={[0.14, 1.05, -0.05]} castShadow>
        <sphereGeometry args={[0.22, 18, 18]} />
        <meshStandardMaterial color="#16a34a" roughness={1} metalness={0} />
      </mesh>
      <mesh position={[-0.12, 1.03, 0.06]} castShadow>
        <sphereGeometry args={[0.24, 18, 18]} />
        <meshStandardMaterial color="#4ade80" roughness={1} metalness={0} />
      </mesh>
    </group>
  );
}

function StreetLamp({
  position,
}: {
  position: [number, number, number];
}): JSX.Element {
  return (
    <group position={position}>
      <mesh position={[0, 0.72, 0]} castShadow>
        <cylinderGeometry args={[0.03, 0.05, 1.42, 10]} />
        <meshStandardMaterial
          color="#94a3b8"
          roughness={0.82}
          metalness={0.12}
        />
      </mesh>
      <mesh position={[0.14, 1.38, 0]} castShadow>
        <sphereGeometry args={[0.08, 12, 12]} />
        <meshStandardMaterial
          color="#fde68a"
          emissive="#fde68a"
          emissiveIntensity={0.9}
          roughness={0.14}
          metalness={0.02}
        />
      </mesh>
      <mesh position={[0.08, 1.2, 0]} rotation={[0, 0, -0.18]} castShadow>
        <boxGeometry args={[0.18, 0.04, 0.04]} />
        <meshStandardMaterial
          color="#cbd5e1"
          roughness={0.8}
          metalness={0.14}
        />
      </mesh>
    </group>
  );
}

function MarketHub({
  position,
}: {
  position: [number, number, number];
}): JSX.Element {
  return (
    <group position={position}>
      <mesh position={[0, 0.28, 0]} castShadow receiveShadow>
        <boxGeometry args={[2.9, 0.56, 2.9]} />
        <meshStandardMaterial
          color="#0f766e"
          roughness={0.66}
          metalness={0.12}
        />
      </mesh>
      <mesh position={[0, 0.72, 0]} castShadow>
        <boxGeometry args={[2.1, 0.14, 2.1]} />
        <meshStandardMaterial
          color="#2dd4bf"
          emissive="#2dd4bf"
          emissiveIntensity={0.24}
          roughness={0.22}
          metalness={0.1}
        />
      </mesh>
      <mesh position={[0, 1.16, 0]} castShadow>
        <cylinderGeometry args={[0.12, 0.14, 0.76, 10]} />
        <meshStandardMaterial
          color="#cbd5e1"
          roughness={0.78}
          metalness={0.16}
        />
      </mesh>
      <mesh position={[0, 1.72, 0]} castShadow>
        <boxGeometry args={[0.66, 0.46, 0.66]} />
        <meshStandardMaterial
          color="#0f172a"
          emissive="#22d3ee"
          emissiveIntensity={0.6}
          roughness={0.26}
          metalness={0.12}
        />
      </mesh>
      <mesh position={[0, 2.18, 0]} castShadow>
        <sphereGeometry args={[0.18, 18, 18]} />
        <meshStandardMaterial
          color="#fde68a"
          emissive="#fde68a"
          emissiveIntensity={1.1}
          roughness={0.1}
          metalness={0.02}
        />
      </mesh>
      <mesh position={[1.05, 0.98, 0]} castShadow>
        <boxGeometry args={[0.16, 1.02, 0.16]} />
        <meshStandardMaterial color="#94a3b8" roughness={0.6} metalness={0.2} />
      </mesh>
      <mesh position={[-1.05, 0.98, 0]} castShadow>
        <boxGeometry args={[0.16, 1.02, 0.16]} />
        <meshStandardMaterial color="#94a3b8" roughness={0.6} metalness={0.2} />
      </mesh>
    </group>
  );
}

function SolarArray({
  position,
  daylight,
}: {
  position: [number, number, number];
  daylight: number;
}): JSX.Element {
  return (
    <group position={position} rotation={[-0.18, Math.PI / 4, 0.06]}>
      {[0, 1, 2].map((index) => (
        <mesh key={index} position={[index * 0.28, 0, 0]} castShadow>
          <boxGeometry args={[0.24, 0.03, 0.42]} />
          <meshStandardMaterial
            color="#1d4ed8"
            emissive="#60a5fa"
            emissiveIntensity={0.24 + daylight * 0.46}
            roughness={0.25}
            metalness={0.5}
          />
        </mesh>
      ))}
    </group>
  );
}

function House({
  house,
  index,
  state,
  daylight,
  hasSolarRoof,
}: {
  house: WorldNode;
  index: number;
  state?: HouseStateSummary;
  daylight: number;
  hasSolarRoof: boolean;
}): JSX.Element {
  const facadeColor = houseColor(index);
  const nightFactor = 1 - daylight;
  const productionGlow = state?.production ?? 0;
  const batteryGlow = state?.battery ?? 0;
  const gridGlow = state?.gridImport ?? 0;
  const tradeGlow = Math.max(state?.p2pBuy ?? 0, state?.p2pSell ?? 0);
  const windowGlow = 0.12 + nightFactor * 0.7 + tradeGlow * 0.18;
  const wallHeight = 1.08 + (index % 3) * 0.06;

  return (
    <group position={house.position}>
      <mesh position={[0, wallHeight / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.72, wallHeight, 1.42]} />
        <meshStandardMaterial
          color={facadeColor}
          roughness={0.84}
          metalness={0.02}
          emissive="#0f172a"
          emissiveIntensity={productionGlow * 0.05}
        />
      </mesh>
      <mesh
        position={[0, wallHeight + 0.34, 0]}
        rotation={[0, Math.PI / 4, 0]}
        castShadow
      >
        <coneGeometry args={[1.14, 0.74, 4]} />
        <meshStandardMaterial
          color="#111827"
          roughness={0.88}
          metalness={0.04}
        />
      </mesh>
      {hasSolarRoof ? (
        <SolarArray
          position={[0.18, wallHeight + 0.48, -0.08]}
          daylight={daylight}
        />
      ) : null}
      <mesh position={[0, 0.32, 0.72]} castShadow>
        <boxGeometry args={[0.34, 0.62, 0.06]} />
        <meshStandardMaterial color="#5b4636" roughness={0.95} metalness={0} />
      </mesh>
      <mesh position={[-0.43, 0.58, 0.74]} castShadow>
        <boxGeometry args={[0.24, 0.24, 0.05]} />
        <meshStandardMaterial
          color="#f8fafc"
          emissive="#fde68a"
          emissiveIntensity={windowGlow}
          roughness={0.16}
          metalness={0.06}
        />
      </mesh>
      <mesh position={[0.43, 0.58, 0.74]} castShadow>
        <boxGeometry args={[0.24, 0.24, 0.05]} />
        <meshStandardMaterial
          color="#f8fafc"
          emissive="#fde68a"
          emissiveIntensity={windowGlow}
          roughness={0.16}
          metalness={0.06}
        />
      </mesh>
      <mesh position={[0, 0.96, 0.74]} castShadow>
        <boxGeometry args={[0.2, 0.18, 0.05]} />
        <meshStandardMaterial
          color="#f8fafc"
          emissive="#fde68a"
          emissiveIntensity={windowGlow * 0.8}
          roughness={0.16}
          metalness={0.06}
        />
      </mesh>
      <mesh position={[0.88, 0.34, -0.08]} castShadow>
        <boxGeometry args={[0.18, 0.52, 0.22]} />
        <meshStandardMaterial
          color="#2dd4bf"
          emissive="#2dd4bf"
          emissiveIntensity={0.16 + batteryGlow * 0.34 + nightFactor * 0.08}
          roughness={0.22}
          metalness={0.08}
        />
      </mesh>
      <mesh position={[-0.88, 0.38, -0.08]} castShadow>
        <sphereGeometry args={[0.1, 14, 14]} />
        <meshStandardMaterial
          color={gridGlow > 0.08 ? "#38bdf8" : "#94a3b8"}
          emissive="#38bdf8"
          emissiveIntensity={gridGlow * 0.55}
          roughness={0.24}
          metalness={0.04}
        />
      </mesh>
      <mesh position={[0, 1.72, 0.46]} castShadow>
        <sphereGeometry args={[0.08, 14, 14]} />
        <meshStandardMaterial
          color={tradeGlow > 0.08 ? "#fbbf24" : "#7c3aed"}
          emissive={tradeGlow > 0.08 ? "#fbbf24" : "#7c3aed"}
          emissiveIntensity={tradeGlow * 0.5}
          roughness={0.18}
          metalness={0.08}
        />
      </mesh>
    </group>
  );
}

function SunRig({
  daylight,
  progress,
}: {
  daylight: number;
  progress: number;
}): JSX.Element {
  const sunRef = useRef<THREE.DirectionalLight>(null);
  const moonRef = useRef<THREE.DirectionalLight>(null);
  const ambientRef = useRef<THREE.AmbientLight>(null);
  const starsMaterialRef = useRef<THREE.PointsMaterial>(null);
  const currentDaylight = useRef(daylight);

  const sunAngle = progress * TAU - Math.PI / 2;
  const sunPosition: [number, number, number] = [
    Math.cos(sunAngle) * 44,
    16 + daylight * 16,
    Math.sin(sunAngle) * 26,
  ];
  const moonPosition: [number, number, number] = [
    Math.cos(sunAngle + Math.PI) * 34,
    15 + (1 - daylight) * 9,
    Math.sin(sunAngle + Math.PI) * 22,
  ];

  useFrame((state, delta) => {
    currentDaylight.current = mix(
      currentDaylight.current,
      daylight,
      clamp(delta * 1.6, 0, 1),
    );
    const liveDaylight = currentDaylight.current;
    const night = 1 - liveDaylight;

    const sky = new THREE.Color("#07111f");
    sky.lerp(new THREE.Color("#93d6ff"), liveDaylight * 0.92);
    state.scene.background = sky;

    if (state.scene.fog) {
      state.scene.fog.color = sky;
    } else {
      state.scene.fog = new THREE.Fog(sky, 34, 110);
    }

    if (sunRef.current) {
      sunRef.current.position.copy(new THREE.Vector3(...sunPosition));
      sunRef.current.intensity = 0.22 + liveDaylight * 3.2;
    }

    if (moonRef.current) {
      moonRef.current.position.copy(new THREE.Vector3(...moonPosition));
      moonRef.current.intensity = 0.12 + night * 0.96;
    }

    if (ambientRef.current) {
      ambientRef.current.intensity = 0.16 + liveDaylight * 0.72;
      ambientRef.current.color = new THREE.Color().lerpColors(
        new THREE.Color("#16263e"),
        new THREE.Color("#fef3c7"),
        liveDaylight * 0.72,
      );
    }

    if (starsMaterialRef.current) {
      starsMaterialRef.current.opacity = 0.05 + night * 0.95;
    }
  });

  const starPositions = useMemo(() => {
    const points: number[] = [];
    for (let index = 0; index < 240; index += 1) {
      const theta = (index / 240) * Math.PI * 2;
      const phi = ((index % 17) / 17) * Math.PI * 0.5 + 0.2;
      const radius = 40 + (index % 7) * 4;
      points.push(
        Math.cos(theta) * radius * Math.sin(phi),
        Math.cos(phi) * radius,
        Math.sin(theta) * radius * Math.sin(phi),
      );
    }
    return new Float32Array(points);
  }, []);

  return (
    <>
      <ambientLight ref={ambientRef} intensity={0.55} color="#ffffff" />
      <hemisphereLight intensity={0.32} color="#c8ecff" groundColor="#0f172a" />
      <directionalLight
        ref={sunRef}
        castShadow
        intensity={2.4}
        color="#fff1c9"
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <directionalLight ref={moonRef} intensity={0.2} color="#b8d6ff" />
      <pointLight position={[-12, 7, -10]} intensity={0.35} color="#67e8f9" />
      <pointLight position={[10, 6, 8]} intensity={0.25} color="#fde68a" />
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[starPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          ref={starsMaterialRef}
          color="#e2e8f0"
          transparent
          opacity={0.12}
          size={0.18}
          sizeAttenuation
        />
      </points>
      <mesh position={[sunPosition[0], sunPosition[1], sunPosition[2]]}>
        <sphereGeometry args={[1.1, 24, 24]} />
        <meshStandardMaterial
          color="#fde68a"
          emissive="#fde68a"
          emissiveIntensity={2.2}
          roughness={0.08}
          metalness={0.02}
        />
      </mesh>
      <mesh position={[moonPosition[0], moonPosition[1], moonPosition[2]]}>
        <sphereGeometry args={[0.72, 24, 24]} />
        <meshStandardMaterial
          color="#dbeafe"
          emissive="#dbeafe"
          emissiveIntensity={1.1}
          roughness={0.12}
          metalness={0.04}
        />
      </mesh>
    </>
  );
}

const TRADE_COLD = new THREE.Color("#2dd4bf"); // teal
const TRADE_HOT = new THREE.Color("#fbbf24"); // amber

function tradeColorFromPrice(params: {
  price: number;
  cda: number;
  min: number;
  max: number;
}): string {
  const { price, cda, min, max } = params;
  const span = Math.max(1e-9, max - min);

  // Piecewise mapping with midpoint at cda (so "below cda" stays teal-ish)
  let t: number;
  if (cda > min + 1e-9 && cda < max - 1e-9) {
    if (price <= cda) {
      t = (0.5 * (price - min)) / Math.max(1e-9, cda - min);
    } else {
      t = 0.5 + (0.5 * (price - cda)) / Math.max(1e-9, max - cda);
    }
  } else {
    // If cda is outside range, just normalize across min..max
    t = (price - min) / span;
  }

  t = clamp(t, 0, 1);
  const c = TRADE_COLD.clone().lerp(TRADE_HOT, t);
  return `#${c.getHexString()}`;
}

function EnergyArcFlow({
  link,
  emphasized = false,
  fade = 1.0,
}: {
  link: FlowLink;
  emphasized?: boolean;
  fade?: number;
}): JSX.Element {
  const arrowRef = useRef<THREE.Mesh>(null);
  const instRef = useRef<THREE.InstancedMesh>(null);

  const source = link.source;
  const target = link.target;

  const dist = Math.hypot(target[0] - source[0], target[2] - source[2]);
  const lift = 0.45 + dist * 0.085;
  const mid: [number, number, number] = [
    (source[0] + target[0]) / 2,
    lift,
    (source[2] + target[2]) / 2,
  ];

  const qty = link.quantity ?? 0.25;

  const widthMul = emphasized ? 1.35 : 1.0;
  const baseWidth = clamp(0.9 + qty * 2.4, 0.9, 3.4) * widthMul;

  const baseOpacity = (emphasized ? 0.75 : 0.35) * fade;
  const pulseOpacity = (emphasized ? 0.95 : 0.75) * fade;

  // number of packets: more qty => more packets
  const packetCount = useMemo(() => {
    const n = Math.round(3 + clamp(qty, 0, 1) * 7); // 3..10
    return clamp(n, 3, 10);
  }, [qty]);

  const curve = useMemo(() => {
    return new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(...source),
      new THREE.Vector3(...mid),
      new THREE.Vector3(...target),
    );
  }, [
    source[0],
    source[1],
    source[2],
    mid[0],
    mid[1],
    mid[2],
    target[0],
    target[1],
    target[2],
  ]);

  // temp objects for instancing
  const tmp = useMemo(() => {
    return {
      m: new THREE.Matrix4(),
      p: new THREE.Vector3(),
      q: new THREE.Quaternion(),
      s: new THREE.Vector3(1, 1, 1),
    };
  }, []);

  useFrame((state) => {
    const t0 =
      (state.clock.elapsedTime * link.pulseSpeed + link.pulsePhase) % 1;

    // packets
    if (instRef.current) {
      for (let i = 0; i < packetCount; i += 1) {
        const tt = (t0 + i / packetCount) % 1;
        const p = curve.getPointAt(tt);

        // slight "sparkle" vertical wobble
        const wobble =
          Math.sin((state.clock.elapsedTime * 3.2 + i * 1.7) * 1.2) * 0.05;

        tmp.p.set(p.x, p.y + wobble, p.z);
        const scale = 0.12 + (emphasized ? 0.05 : 0.0);
        tmp.s.setScalar(scale);

        tmp.m.compose(tmp.p, tmp.q, tmp.s);
        instRef.current.setMatrixAt(i, tmp.m);
      }
      instRef.current.instanceMatrix.needsUpdate = true;
    }

    // arrowhead at the end
    if (arrowRef.current) {
      const t2 = 0.985;
      const p2 = curve.getPointAt(t2);
      const tangent = curve.getTangentAt(t2).normalize();

      arrowRef.current.position.copy(p2);
      const up = new THREE.Vector3(0, 1, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(up, tangent);
      arrowRef.current.quaternion.copy(quat);
    }
  });

  return (
    <group>
      <QuadraticBezierLine
        start={source}
        end={target}
        mid={mid}
        color={link.color}
        lineWidth={baseWidth}
        transparent
        opacity={baseOpacity}
      />

      {/* packet train (instanced) */}
      <instancedMesh
        ref={instRef}
        args={[undefined as any, undefined as any, packetCount]}
      >
        <sphereGeometry args={[1, 12, 12]} />
        <meshStandardMaterial
          color={link.color}
          emissive={link.color}
          emissiveIntensity={1.25}
          transparent
          opacity={pulseOpacity}
          roughness={0.25}
        />
      </instancedMesh>

      {/* arrowhead */}
      <mesh ref={arrowRef}>
        <coneGeometry args={[0.12 + (emphasized ? 0.06 : 0), 0.3, 12]} />
        <meshStandardMaterial
          color={link.color}
          emissive={link.color}
          emissiveIntensity={0.8}
          transparent
          opacity={0.85 * fade}
          roughness={0.25}
        />
      </mesh>
    </group>
  );
}

function linkInvolvesHouse(link: FlowLink, houseId: number): boolean {
  if (link.kind === "trade") {
    return link.buyerId === houseId || link.sellerId === houseId;
  }
  if (link.kind === "order" || link.kind === "grid") {
    return link.householdId === houseId;
  }
  return false;
}

function WindTurbine({
  position,
  spinRps = 0.8,
  scale = 1.0,
}: {
  position: [number, number, number];
  spinRps?: number; // rotations per second
  scale?: number;
}): JSX.Element {
  const rotorRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!rotorRef.current) return;
    rotorRef.current.rotation.x = state.clock.elapsedTime * spinRps * TAU;
  });

  return (
    <group position={position} scale={[scale, scale, scale]}>
      {/* concrete pad */}
      <mesh position={[0, 0.04, 0]} receiveShadow>
        <cylinderGeometry args={[0.8, 0.9, 0.08, 20]} />
        <meshStandardMaterial
          color="#475569"
          roughness={0.95}
          metalness={0.05}
        />
      </mesh>

      {/* tower */}
      <mesh position={[0, 1.6, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.12, 0.22, 3.2, 16]} />
        <meshStandardMaterial
          color="#cbd5e1"
          roughness={0.75}
          metalness={0.12}
        />
      </mesh>

      {/* nacelle */}
      <mesh position={[0, 3.25, 0]} castShadow>
        <boxGeometry args={[0.9, 0.32, 0.36]} />
        <meshStandardMaterial
          color="#e2e8f0"
          roughness={0.45}
          metalness={0.18}
        />
      </mesh>

      {/* rotor + blades (spin) */}
      {/* hub */}
      <mesh position={[0.52, 3.25, 0]} castShadow>
        <sphereGeometry args={[0.14, 18, 18]} />
        <meshStandardMaterial
          color="#f8fafc"
          roughness={0.3}
          metalness={0.15}
        />
      </mesh>

      {/* rotor spins about X axis */}
      <group ref={rotorRef} position={[0.52, 3.25, 0]}>
        {[0, 1, 2].map((i) => {
          const ang = (TAU * i) / 3;
          const bladeLength = 1.6;

          return (
            <group key={i} rotation={[ang, 0, 0]}>
              <mesh position={[0, bladeLength / 2, 0]} castShadow>
                <boxGeometry args={[0.08, bladeLength, 0.14]} />
                <meshStandardMaterial
                  color="#f8fafc"
                  roughness={0.35}
                  metalness={0.08}
                />
              </mesh>
            </group>
          );
        })}
      </group>
    </group>
  );
}

function GroundSolarFarm({
  position,
  daylight,
  scale = 1.0,
}: {
  position: [number, number, number];
  daylight: number;
  scale?: number;
}): JSX.Element {
  const glow = 0.12 + daylight * 0.65;

  return (
    <group position={position} scale={[scale, scale, scale]}>
      {/* pad */}
      <mesh position={[0, 0.02, 0]} receiveShadow>
        <boxGeometry args={[2.8, 0.04, 2.2]} />
        <meshStandardMaterial
          color="#334155"
          roughness={0.95}
          metalness={0.05}
        />
      </mesh>

      {/* racks: 2 rows × 3 panels */}
      {Array.from({ length: 2 }, (_, row) =>
        Array.from({ length: 3 }, (_, col) => {
          const x = (col - 1) * 0.82;
          const z = (row - 0.5) * 0.72;
          return (
            <group
              key={`${row}-${col}`}
              position={[x, 0.08, z]}
              rotation={[-0.25, Math.PI / 4, 0]}
            >
              <mesh castShadow>
                <boxGeometry args={[0.68, 0.04, 0.44]} />
                <meshStandardMaterial
                  color="#1d4ed8"
                  emissive="#60a5fa"
                  emissiveIntensity={glow}
                  roughness={0.25}
                  metalness={0.5}
                />
              </mesh>
              {/* support */}
              <mesh position={[0, -0.12, 0]} castShadow>
                <boxGeometry args={[0.72, 0.03, 0.06]} />
                <meshStandardMaterial
                  color="#64748b"
                  roughness={0.9}
                  metalness={0.12}
                />
              </mesh>
            </group>
          );
        }),
      )}

      {/* inverter box */}
      <mesh position={[1.25, 0.22, 0.9]} castShadow>
        <boxGeometry args={[0.34, 0.42, 0.22]} />
        <meshStandardMaterial
          color="#0f172a"
          emissive="#22d3ee"
          emissiveIntensity={0.08 + daylight * 0.18}
          roughness={0.7}
          metalness={0.15}
        />
      </mesh>
    </group>
  );
}
function summarizeHouseTrades(houseId: number, trades: P2PTrade[]) {
  let boughtQty = 0;
  let soldQty = 0;
  let spent = 0;
  let earned = 0;

  for (const t of trades) {
    if (t.buyer_household_id === houseId) {
      boughtQty += t.quantity;
      spent += t.quantity * t.price;
    }
    if (t.seller_household_id === houseId) {
      soldQty += t.quantity;
      earned += t.quantity * t.price;
    }
  }

  return {
    boughtQty,
    soldQty,
    avgBuyPrice: boughtQty > 1e-9 ? spent / boughtQty : 0,
    avgSellPrice: soldQty > 1e-9 ? earned / soldQty : 0,
    tradeCount: trades.filter(
      (t) =>
        t.buyer_household_id === houseId || t.seller_household_id === houseId,
    ).length,
  };
}

function fmt(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "0.00";
  return value.toFixed(digits);
}

function buildCounterpartySummary(houseId: number, trades: P2PTrade[]) {
  // counterpartyId -> { qty, value }
  const map = new Map<number, { qty: number; value: number }>();

  for (const t of trades) {
    const isBuyer = t.buyer_household_id === houseId;
    const isSeller = t.seller_household_id === houseId;
    if (!isBuyer && !isSeller) continue;

    const counterparty = isBuyer ? t.seller_household_id : t.buyer_household_id;
    const entry = map.get(counterparty) ?? { qty: 0, value: 0 };
    entry.qty += t.quantity;
    entry.value += t.quantity * t.price;
    map.set(counterparty, entry);
  }

  return [...map.entries()]
    .map(([id, v]) => ({
      id,
      qty: v.qty,
      vwap: v.qty > 1e-9 ? v.value / v.qty : 0,
    }))
    .sort((a, b) => b.qty - a.qty)
    .slice(0, 5);
}

function HouseTooltip({
  house,
  raw,
  trades,
}: {
  house: WorldNode;
  raw?: HouseStateRaw;
  trades: P2PTrade[];
}): JSX.Element {
  const stats = summarizeHouseTrades(house.id, trades);
  const counterparties = buildCounterpartySummary(house.id, trades);

  const net = raw?.netBalance ?? 0;
  const netLabel = net >= 0 ? `+${fmt(net, 2)}` : fmt(net, 2);

  return (
    <Html
      position={[house.position[0], 3.6, house.position[2]]} // a bit higher
      center
      transform
      sprite
      distanceFactor={10}
      zIndexRange={[100, 0]}
      style={{ pointerEvents: "none" }}
    >
      <div
        style={{
          width: 270,
          padding: 10,
          borderRadius: 12,
          background: "rgba(15,23,42,0.92)",
          border: "1px solid rgba(148,163,184,0.35)",
          color: "#e2e8f0",
          fontSize: 12,
          lineHeight: 1.25,
        }}
      >
        <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>
          {house.label ?? `House ${house.id}`}
        </div>

        <div
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}
        >
          <div>Prod: {fmt(raw?.production ?? 0)} kW</div>
          <div>Cons: {fmt(raw?.consumption ?? 0)} kW</div>
          <div>Battery: {fmt(raw?.batteryLevel ?? 0)} kWh</div>
          <div>Grid import: {fmt(raw?.gridImport ?? 0)} kW</div>
          <div>P2P buy: {fmt(raw?.p2pBuy ?? 0)} kW</div>
          <div>P2P sell: {fmt(raw?.p2pSell ?? 0)} kW</div>
          <div>Price: {fmt(raw?.price ?? 0, 3)}</div>
          <div>Net: {netLabel}</div>
        </div>

        <div style={{ marginTop: 8, opacity: 0.92 }}>
          Trades: {stats.tradeCount}
          <br />
          Bought: {fmt(stats.boughtQty)} @ {fmt(stats.avgBuyPrice, 3)}
          <br />
          Sold: {fmt(stats.soldQty)} @ {fmt(stats.avgSellPrice, 3)}
        </div>

        {counterparties.length ? (
          <div style={{ marginTop: 8, opacity: 0.85 }}>
            Top counterparties:
            <div style={{ marginTop: 6 }}>
              {counterparties.map((c) => (
                <div key={c.id}>
                  #{c.id}: {fmt(c.qty)} @ {fmt(c.vwap, 3)}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div style={{ marginTop: 8, opacity: 0.72 }}>
          Click to pin • click ground to clear
        </div>
      </div>
    </Html>
  );
}

function HouseMarker({
  position,
  color,
  visible,
}: {
  position: [number, number, number];
  color: string;
  visible: boolean;
}): JSX.Element | null {
  if (!visible) return null;

  return (
    <group position={position}>
      {/* ground ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
        <ringGeometry args={[0.9, 1.18, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.7}
          transparent
          opacity={0.9}
          roughness={0.2}
        />
      </mesh>

      {/* vertical beacon */}
      <mesh position={[0, 1.7, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 3.0, 12]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.2}
          transparent
          opacity={0.25}
        />
      </mesh>
    </group>
  );
}

function PickableHouse({
  house,
  index,
  viz,
  raw,
  daylight,
  hasSolarRoof,
  trades,
  hovered,
  selected,
  onHover,
  onUnhover,
  onToggleSelect,
}: {
  house: WorldNode;
  index: number;
  viz?: HouseStateSummary;
  raw?: HouseStateRaw;
  daylight: number;
  hasSolarRoof: boolean;
  trades: P2PTrade[];
  hovered: boolean;
  selected: boolean;
  onHover: () => void;
  onUnhover: () => void;
  onToggleSelect: () => void;
}): JSX.Element {
  useCursor(hovered || selected);

  return (
    <group
      onPointerOver={(e) => {
        e.stopPropagation();
        onHover();
      }}
      onPointerOut={(e) => {
        e.stopPropagation();
        onUnhover();
      }}
      onClick={(e) => {
        e.stopPropagation();
        onToggleSelect();
      }}
    >
      <House
        house={house}
        index={index}
        state={viz}
        daylight={daylight}
        hasSolarRoof={hasSolarRoof}
      />

      <HouseMarker
        position={house.position}
        color={selected ? "#fbbf24" : "#22d3ee"}
        visible={hovered || selected}
      />

      {(hovered || selected) && (
        <HouseTooltip house={house} raw={raw} trades={trades} />
      )}
    </group>
  );
}

function ControlsRig({
  hoveredHouseId,
  selectedHouseId,
  houseById,
}: {
  hoveredHouseId: number | null;
  selectedHouseId: number | null;
  houseById: Map<number, WorldNode>;
}): JSX.Element {
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const { camera } = useThree();

  const desiredTargetRef = useRef<THREE.Vector3 | null>(null);
  const desiredCameraRef = useRef<THREE.Vector3 | null>(null);

  // Pause auto rotate on any hover/selection
  const paused = hoveredHouseId !== null || selectedHouseId !== null;

  useEffect(() => {
    if (!selectedHouseId) {
      desiredTargetRef.current = null;
      desiredCameraRef.current = null;
      return;
    }

    const node = houseById.get(selectedHouseId);
    if (!node) return;

    const controls = controlsRef.current;
    const currentTarget =
      controls?.target?.clone() ?? new THREE.Vector3(0, 0, 0);

    const desiredTarget = new THREE.Vector3(
      node.position[0],
      0.9,
      node.position[2],
    );

    // keep same viewing offset, just move the orbit center
    const offset = camera.position.clone().sub(currentTarget);
    const desiredCamera = desiredTarget.clone().add(offset);

    desiredTargetRef.current = desiredTarget;
    desiredCameraRef.current = desiredCamera;
  }, [selectedHouseId, houseById, camera]);

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (!controls) return;

    controls.autoRotate = !paused;
    controls.autoRotateSpeed = 0.18;

    const desiredTarget = desiredTargetRef.current;
    const desiredCamera = desiredCameraRef.current;

    if (desiredTarget && desiredCamera) {
      const t = clamp(delta * 2.4, 0, 1);
      controls.target.lerp(desiredTarget, t);
      camera.position.lerp(desiredCamera, t);
      controls.update();
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enablePan={false}
      minDistance={24}
      maxDistance={72}
      minPolarAngle={0.42}
      maxPolarAngle={1.22}
      autoRotate={!paused}
      autoRotateSpeed={0.18}
    />
  );
}

function MarketHUD({
  hubPosition,
  marketSnapshot,
  selectedHouseId,
  showGridFlows,
  setShowGridFlows,
  showOrderFlows,
  setShowOrderFlows,
  showTradeFlows,
  setShowTradeFlows,
  maxFlows,
  setMaxFlows,
}: {
  hubPosition: [number, number, number];
  marketSnapshot?: Record<string, unknown>;
  selectedHouseId: number | null;

  showGridFlows: boolean;
  setShowGridFlows: (v: boolean) => void;
  showOrderFlows: boolean;
  setShowOrderFlows: (v: boolean) => void;
  showTradeFlows: boolean;
  setShowTradeFlows: (v: boolean) => void;

  maxFlows: number;
  setMaxFlows: (v: number) => void;
}): JSX.Element {
  const clearing = asFiniteNumber(marketSnapshot?.["clearing_price"]) ?? 0;
  const cda = asFiniteNumber(marketSnapshot?.["p2p_cda_price"]) ?? 0;
  const p2pVol = asFiniteNumber(marketSnapshot?.["p2p_traded_volume"]) ?? 0;

  const trades = parseP2PTrades(marketSnapshot);
  const tradeCount = trades.length;
  const demo = useSceneDirectorStore((s) => s.demo);

  return (
    <Html
      position={[hubPosition[0], 3.4, hubPosition[2]]}
      transform
      sprite
      center
      distanceFactor={10}
      style={{ pointerEvents: "auto" }}
    >
      <div
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 290,
          padding: 10,
          borderRadius: 12,
          background: "rgba(2,6,23,0.84)",
          border: "1px solid rgba(148,163,184,0.32)",
          color: "#e2e8f0",
          fontSize: 12,
        }}
      >
        <div style={{ marginTop: 8, opacity: 0.85 }}>
          Demo: <span style={{ fontWeight: 700 }}>{demo.phase}</span>{" "}
          <span style={{ opacity: 0.75 }}>
            ({Math.round(demo.progress * 100)}%)
          </span>
        </div>
        <div style={{ fontWeight: 800, marginBottom: 8 }}>
          Market Hub
          {selectedHouseId ? (
            <span style={{ opacity: 0.75 }}>
              {" "}
              • selected #{selectedHouseId}
            </span>
          ) : null}
        </div>

        <div
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}
        >
          <div>Clearing: {fmt(clearing, 3)}</div>
          <div>P2P CDA: {fmt(cda, 3)}</div>
          <div>P2P vol: {fmt(p2pVol, 3)}</div>
          <div># trades: {tradeCount}</div>
        </div>

        <div style={{ marginTop: 10, fontWeight: 700, opacity: 0.9 }}>
          Flow layers
        </div>

        <div
          style={{ display: "flex", gap: 10, marginTop: 6, flexWrap: "wrap" }}
        >
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showGridFlows}
              onChange={(e) => setShowGridFlows(e.target.checked)}
            />
            Grid
          </label>
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showOrderFlows}
              onChange={(e) => setShowOrderFlows(e.target.checked)}
            />
            Orders
          </label>
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showTradeFlows}
              onChange={(e) => setShowTradeFlows(e.target.checked)}
            />
            Trades
          </label>
        </div>

        <div style={{ marginTop: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontWeight: 700, opacity: 0.9 }}>Max flows</span>
            <span style={{ opacity: 0.75 }}>{maxFlows}</span>
          </div>
          <input
            type="range"
            min={10}
            max={120}
            value={maxFlows}
            onChange={(e) => setMaxFlows(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </div>
      </div>
    </Html>
  );
}

function makeGrassTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(size, size);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const i = (y * size + x) * 4;

      // simple layered noise
      const n1 = Math.random();
      const n2 = Math.random();
      const n = 0.65 * n1 + 0.35 * n2;

      // grass palette
      const r = Math.floor(18 + n * 22);
      const g = Math.floor(40 + n * 65);
      const b = Math.floor(18 + n * 26);

      img.data[i + 0] = r;
      img.data[i + 1] = g;
      img.data[i + 2] = b;
      img.data[i + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(10, 10);
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

function Terrain(): JSX.Element {
  const tex = useMemo(() => makeGrassTexture(), []);

  const geom = useMemo(() => {
    const g = new THREE.PlaneGeometry(420, 320, 96, 96);
    const pos = g.attributes.position;
    const v = new THREE.Vector3();

    // create gentle hills toward edges, keep center mostly flat
    for (let i = 0; i < pos.count; i += 1) {
      v.fromBufferAttribute(pos, i);
      const r = Math.hypot(v.x, v.y);
      const edge = clamp((r - 70) / 120, 0, 1); // near center ~0, edges ~1
      const hills =
        (Math.sin(v.x * 0.08) + Math.cos(v.y * 0.07)) * 0.6 +
        Math.sin(v.x * 0.16 + v.y * 0.11) * 0.35;

      // displace along Z (since plane is XY before rotation)
      v.z = edge * hills * 1.2;
      pos.setXYZ(i, v.x, v.y, v.z);
    }

    g.computeVertexNormals();
    return g;
  }, []);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
      <primitive object={geom} />
      <meshStandardMaterial map={tex} roughness={1} metalness={0} />
    </mesh>
  );
}

function DistrictPad(): JSX.Element {
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, -0.02, 0]}
      receiveShadow
    >
      <planeGeometry args={[120, 90]} />
      <meshStandardMaterial color="#0b1220" roughness={1} metalness={0} />
    </mesh>
  );
}

function makeWindowTexture(): THREE.CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#050814";
  ctx.fillRect(0, 0, size, size);

  // simple window grid
  const cellW = 10;
  const cellH = 12;
  for (let y = 0; y < size; y += cellH) {
    for (let x = 0; x < size; x += cellW) {
      const lit = Math.random() > 0.72;
      ctx.fillStyle = lit ? "rgba(253,230,138,0.85)" : "rgba(15,23,42,0.06)";
      ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 5);
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1, 1);
  tex.anisotropy = 8;
  return tex;
}

function DistantSkyline({
  seed,
  daylight,
}: {
  seed: number;
  daylight: number;
}): JSX.Element {
  const windowTex = useMemo(() => makeWindowTexture(), []);
  const rand = useMemo(() => mulberry32(seed ^ 0x9e3779b9), [seed]);

  const instances = useMemo(() => {
    const items: {
      pos: THREE.Vector3;
      scale: THREE.Vector3;
      rotY: number;
      color: THREE.Color;
    }[] = [];

    // ring around district/terrain
    const ringR = 175;
    const count = 120;

    for (let i = 0; i < count; i += 1) {
      const ang = (i / count) * TAU;
      const jitterR = ringR + (rand() - 0.5) * 26;
      const x = Math.cos(ang) * jitterR;
      const z = Math.sin(ang) * jitterR;

      const w = 5 + rand() * 10;
      const d = 5 + rand() * 10;
      const h = 10 + rand() * 46;

      const rotY = rand() * TAU;

      const base = new THREE.Color("#0b1220");
      base.lerp(new THREE.Color("#111827"), rand() * 0.6);

      items.push({
        pos: new THREE.Vector3(x, h / 2 - 0.06, z),
        scale: new THREE.Vector3(w, h, d),
        rotY,
        color: base,
      });
    }

    return items;
  }, [rand]);

  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useMemo(() => {
    if (!meshRef.current) return;
    instances.forEach((it, i) => {
      dummy.position.copy(it.pos);
      dummy.rotation.set(0, it.rotY, 0);
      dummy.scale.copy(it.scale);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
      meshRef.current!.setColorAt(i, it.color);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor)
      meshRef.current.instanceColor.needsUpdate = true;
  }, [instances, dummy]);

  // windows more visible at night
  const night = 1 - daylight;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined as any, undefined as any, instances.length]}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        vertexColors
        map={windowTex}
        emissiveMap={windowTex}
        emissive="#fde68a"
        emissiveIntensity={0.08 + night * 0.65}
        roughness={0.92}
        metalness={0.02}
      />
    </instancedMesh>
  );
}

function OutskirtsTrees({ seed }: { seed: number }): JSX.Element {
  const rand = useMemo(() => mulberry32(seed ^ 0x1234567), [seed]);

  const count = 520;

  const trunksRef = useRef<THREE.InstancedMesh>(null);
  const canopiesRef = useRef<THREE.InstancedMesh>(null);

  const dummy = useMemo(() => new THREE.Object3D(), []);

  const points = useMemo(() => {
    const pts: { x: number; z: number; s: number }[] = [];
    for (let i = 0; i < count; i += 1) {
      // distribute outside district pad but within terrain
      const x = (rand() - 0.5) * 320;
      const z = (rand() - 0.5) * 240;

      // keep a “clear” region for the neighborhood
      if (Math.abs(x) < 70 && Math.abs(z) < 55) {
        i -= 1;
        continue;
      }

      const s = 0.8 + rand() * 1.6;
      pts.push({ x, z, s });
    }
    return pts;
  }, [rand]);

  useMemo(() => {
    if (!trunksRef.current || !canopiesRef.current) return;

    points.forEach((p, i) => {
      // trunk
      dummy.position.set(p.x, 0.55 * p.s, p.z);
      dummy.rotation.set(0, rand() * TAU, 0);
      dummy.scale.set(1, p.s, 1);
      dummy.updateMatrix();
      trunksRef.current!.setMatrixAt(i, dummy.matrix);

      // canopy
      dummy.position.set(p.x, 1.25 * p.s, p.z);
      dummy.rotation.set(0, rand() * TAU, 0);
      dummy.scale.setScalar(0.8 * p.s);
      dummy.updateMatrix();
      canopiesRef.current!.setMatrixAt(i, dummy.matrix);
    });

    trunksRef.current.instanceMatrix.needsUpdate = true;
    canopiesRef.current.instanceMatrix.needsUpdate = true;
  }, [points, dummy, rand]);

  return (
    <group>
      <instancedMesh
        ref={trunksRef}
        args={[undefined as any, undefined as any, points.length]}
      >
        <cylinderGeometry args={[0.08, 0.11, 1.1, 8]} />
        <meshStandardMaterial color="#5b3a24" roughness={1} metalness={0} />
      </instancedMesh>

      <instancedMesh
        ref={canopiesRef}
        args={[undefined as any, undefined as any, points.length]}
      >
        <sphereGeometry args={[0.55, 10, 10]} />
        <meshStandardMaterial color="#1f8a3b" roughness={1} metalness={0} />
      </instancedMesh>
    </group>
  );
}

function LotGrassPatches({ houses }: { houses: WorldNode[] }): JSX.Element {
  const ref = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useMemo(() => {
    if (!ref.current) return;
    houses.forEach((h, i) => {
      dummy.position.set(h.position[0], 0.001, h.position[2]);
      dummy.rotation.set(-Math.PI / 2, 0, 0);
      dummy.scale.set(3.2, 2.8, 1);
      dummy.updateMatrix();
      ref.current!.setMatrixAt(i, dummy.matrix);
    });
    ref.current.instanceMatrix.needsUpdate = true;
  }, [houses, dummy]);

  return (
    <instancedMesh
      ref={ref}
      args={[undefined as any, undefined as any, houses.length]}
      receiveShadow
    >
      <planeGeometry args={[1, 1]} />
      <meshStandardMaterial color="#12351a" roughness={1} metalness={0} />
    </instancedMesh>
  );
}

function nearestValue(values: number[], v: number): number | undefined {
  if (!values.length) return undefined;
  let best = values[0];
  let bestD = Math.abs(values[0] - v);
  for (let i = 1; i < values.length; i += 1) {
    const d = Math.abs(values[i] - v);
    if (d < bestD) {
      bestD = d;
      best = values[i];
    }
  }
  return best;
}

function Driveways({
  houses,
  roadPlan,
  sidewalkWidth,
}: {
  houses: WorldNode[];
  roadPlan: RoadPlan;
  sidewalkWidth: number;
}): JSX.Element {
  const driveColor = "#1f2937";

  return (
    <group>
      {houses.map((h) => {
        const hx = h.position[0];
        const hz = h.position[2];

        const nearX = nearestValue(roadPlan.verticalXs, hx);
        const nearZ = nearestValue(roadPlan.horizontalZs, hz);

        if (nearX === undefined || nearZ === undefined) return null;

        const dx = Math.abs(nearX - hx);
        const dz = Math.abs(nearZ - hz);

        const roadEdge = ROAD_WIDTH / 2 + sidewalkWidth + 0.18;

        // choose driveway to closest road direction
        let tx = hx;
        let tz = hz;
        if (dx < dz) {
          const dir = Math.sign(hx - nearX) || 1;
          tx = nearX + dir * roadEdge;
          tz = hz;
        } else {
          const dir = Math.sign(hz - nearZ) || 1;
          tz = nearZ + dir * roadEdge;
          tx = hx;
        }

        const len = Math.hypot(tx - hx, tz - hz);
        if (len < 0.4) return null;

        const midX = (hx + tx) / 2;
        const midZ = (hz + tz) / 2;
        const rotY = Math.atan2(tx - hx, tz - hz);

        return (
          <mesh
            key={`drv-${h.id}`}
            position={[midX, 0.012, midZ]}
            rotation={[-Math.PI / 2, rotY, 0]}
            receiveShadow
          >
            <planeGeometry args={[0.7, len]} />
            <meshStandardMaterial
              color={driveColor}
              roughness={0.95}
              metalness={0.02}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function Curbs({
  roadPlan,
  verticalRoadLength,
  horizontalRoadLength,
  roadY,
}: {
  roadPlan: RoadPlan;
  verticalRoadLength: number;
  horizontalRoadLength: number;
  roadY: number;
}): JSX.Element {
  const curbH = 0.035;
  const curbW = 0.06;
  const offset = ROAD_WIDTH / 2 + curbW / 2;

  return (
    <group>
      {roadPlan.verticalXs.flatMap((x) => {
        const left: [number, number, number] = [
          x - offset,
          roadY + curbH / 2,
          0,
        ];
        const right: [number, number, number] = [
          x + offset,
          roadY + curbH / 2,
          0,
        ];

        return [
          <mesh key={`cv-l-${x}`} position={left} receiveShadow>
            <boxGeometry args={[curbW, curbH, verticalRoadLength]} />
            <meshStandardMaterial color="#9aa3b2" roughness={0.9} />
          </mesh>,
          <mesh key={`cv-r-${x}`} position={right} receiveShadow>
            <boxGeometry args={[curbW, curbH, verticalRoadLength]} />
            <meshStandardMaterial color="#9aa3b2" roughness={0.9} />
          </mesh>,
        ];
      })}

      {roadPlan.horizontalZs.flatMap((z) => {
        const top: [number, number, number] = [
          0,
          roadY + curbH / 2,
          z - offset,
        ];
        const bottom: [number, number, number] = [
          0,
          roadY + curbH / 2,
          z + offset,
        ];

        return [
          <mesh key={`ch-t-${z}`} position={top} receiveShadow>
            <boxGeometry args={[horizontalRoadLength, curbH, curbW]} />
            <meshStandardMaterial color="#9aa3b2" roughness={0.9} />
          </mesh>,
          <mesh key={`ch-b-${z}`} position={bottom} receiveShadow>
            <boxGeometry args={[horizontalRoadLength, curbH, curbW]} />
            <meshStandardMaterial color="#9aa3b2" roughness={0.9} />
          </mesh>,
        ];
      })}
    </group>
  );
}

function SceneGraph({
  topology,
  observation,
  latestInfo,
  step,
}: Neighborhood3DSceneProps): JSX.Element {
  const bounds = useMemo(() => getBounds(topology), [topology]);
  const nodes = useMemo(() => topology?.nodes ?? [], [topology]);
  const [hoveredHouseId, setHoveredHouseId] = useState<number | null>(null);
  const selectedHouseId = useSceneDirectorStore((s) => s.selectedHouseId);
  const setSelectedHouseId = useSceneDirectorStore((s) => s.setSelectedHouseId);

  const showGridFlows = useSceneDirectorStore((s) => s.showGridFlows);
  const setShowGridFlows = useSceneDirectorStore((s) => s.setShowGridFlows);

  const showOrderFlows = useSceneDirectorStore((s) => s.showOrderFlows);
  const setShowOrderFlows = useSceneDirectorStore((s) => s.setShowOrderFlows);

  const showTradeFlows = useSceneDirectorStore((s) => s.showTradeFlows);
  const setShowTradeFlows = useSceneDirectorStore((s) => s.setShowTradeFlows);

  const maxFlows = useSceneDirectorStore((s) => s.maxFlows);
  const setMaxFlows = useSceneDirectorStore((s) => s.setMaxFlows);

  const demo = useSceneDirectorStore((s) => s.demo);

  const sceneSeed =
    (getLatestNumeric(latestInfo, ["seed", "episode_id", "episode_count"]) ??
      42) | 0;

  const marketSnapshot = useMemo(
    () => getMarketSnapshotObject(latestInfo),
    [latestInfo],
  );
  const trades = useMemo(
    () => parseP2PTrades(marketSnapshot),
    [marketSnapshot],
  );
  const positionedNodes = useMemo<WorldNode[]>(
    () =>
      nodes.map((node, orderIndex) => ({
        ...node,
        position: toWorldPosition(node, bounds),
        orderIndex,
      })),
    [bounds, nodes],
  );
  const households = useMemo(
    () => positionedNodes.filter((node) => node.type === "household"),
    [positionedNodes],
  );
  const houseById = useMemo(() => {
    const map = new Map<number, WorldNode>();
    for (const h of households) map.set(h.id, h);
    return map;
  }, [households]);
  const solarNodes = useMemo(
    () => positionedNodes.filter((node) => node.type === "solar"),
    [positionedNodes],
  );
  const windNodes = useMemo(
    () => positionedNodes.filter((node) => node.type === "wind"),
    [positionedNodes],
  );
  const gridNode = useMemo(
    () =>
      positionedNodes.find((node) => node.type === "grid") ??
      positionedNodes.find((node) => node.asset_class === "grid"),
    [positionedNodes],
  );
  const hubPosition = gridNode?.position ?? [0, 0, 0];
  const cycle = useMemo(
    () => resolveDayCycle(latestInfo, step),
    [latestInfo, step],
  );
  const houseStateMap = useMemo(
    () => buildHouseStateMap(households, observation),
    [households, observation],
  );

  const houseRawStateMap = useMemo(
    () => buildHouseRawStateMap(households, observation),
    [households, observation],
  );

  const solarByHouseId = useMemo(() => {
    const map = new Map<number, WorldNode[]>();
    for (const s of solarNodes) {
      if (typeof s.serves_household_id === "number") {
        const arr = map.get(s.serves_household_id) ?? [];
        arr.push(s);
        map.set(s.serves_household_id, arr);
      }
    }
    return map;
  }, [solarNodes]);

  const solarRoofSet = useMemo(() => {
    const assigned = new Set<number>();
    households.forEach((house) => {
      if (solarByHouseId.has(house.id)) {
        assigned.add(house.id);
      }
    });
    return assigned;
  }, [households, solarByHouseId]);

  const flowLinks = useMemo(
    () =>
      buildFlowLinks(households, houseStateMap, hubPosition, marketSnapshot),
    [houseStateMap, households, hubPosition, marketSnapshot],
  );

  // const focusedLinks = useMemo(() => {
  //   if (!selectedHouseId) return flowLinks;
  //   return flowLinks.filter(
  //     (l) => l.buyerId === selectedHouseId || l.sellerId === selectedHouseId,
  //   );
  // }, [flowLinks, selectedHouseId]);

  const displayLinks = useMemo(() => {
    let links = flowLinks;

    links = links.filter((l) => {
      if (l.kind === "grid") return showGridFlows;
      if (l.kind === "order") return showOrderFlows;
      if (l.kind === "trade") return showTradeFlows;
      return true;
    });

    // show most important links first
    links = [...links].sort((a, b) => (b.quantity ?? 0) - (a.quantity ?? 0));

    return links.slice(0, clamp(maxFlows, 5, 120));
  }, [flowLinks, showGridFlows, showOrderFlows, showTradeFlows, maxFlows]);

  const districtScale = 1.04 + clamp(positionedNodes.length / 220, 0, 0.18);

  const roadSpanX = Math.max(52, WORLD_WIDTH + 14);
  const roadSpanZ = Math.max(38, WORLD_DEPTH + 12);
  const sideSpanX = roadSpanX + 10;
  const sideSpanZ = roadSpanZ + 8;
  // const roadY = ROAD_THICKNESS / 2;
  // const sidewalkY = ROAD_THICKNESS + SIDEWALK_THICKNESS / 2;

  const roadPlan = useMemo(
    () => deriveRoadPlanFromGrid(households),
    [households],
  );

  const roadY = ROAD_THICKNESS / 2;
  const sidewalkY = ROAD_THICKNESS + SIDEWALK_THICKNESS / 2;
  const sidewalkWidth = 0.28; // visual width around roads

  const verticalRoadLength = roadPlan.spanZ + 4;
  const horizontalRoadLength = roadPlan.spanX + 4;

  const windSpeed = getLatestNumeric(latestInfo, ["wind_speed"]) ?? 0;
  const windFactor = clamp(normalizeSignal(windSpeed), 0, 1);
  const windSpin = 0.35 + windFactor * 1.8;

  return (
    <>
      <Terrain />
      <DistantSkyline seed={sceneSeed} daylight={cycle.daylight} />
      <OutskirtsTrees seed={sceneSeed} />
      <group scale={[districtScale, districtScale, districtScale]}>
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, -0.04, 0]}
          onClick={(e) => {
            e.stopPropagation();
            setSelectedHouseId(null);
          }}
        >
          <planeGeometry args={[120, 90]} />
          <meshBasicMaterial transparent opacity={0} />
        </mesh>
        <DistrictPad />
        <LotGrassPatches houses={households} />
        <Driveways
          houses={households}
          roadPlan={roadPlan}
          sidewalkWidth={sidewalkWidth}
        />
        <Curbs
          roadPlan={roadPlan}
          verticalRoadLength={verticalRoadLength}
          horizontalRoadLength={horizontalRoadLength}
          roadY={roadY}
        />
        {roadPlan.verticalXs.map((x) => (
          <group key={`vroad-${x}`}>
            <Road
              position={[x, roadY, 0]}
              size={[ROAD_WIDTH, ROAD_THICKNESS, verticalRoadLength]}
            />
            <Sidewalk
              position={[x, sidewalkY, 0]}
              size={[
                ROAD_WIDTH + sidewalkWidth * 2,
                SIDEWALK_THICKNESS,
                verticalRoadLength + 0.6,
              ]}
            />
          </group>
        ))}
        {roadPlan.horizontalZs.map((z) => (
          <group key={`hroad-${z}`}>
            <Road
              position={[0, roadY, z]}
              size={[horizontalRoadLength, ROAD_THICKNESS, ROAD_WIDTH]}
            />
            <Sidewalk
              position={[0, sidewalkY, z]}
              size={[
                horizontalRoadLength + 0.6,
                SIDEWALK_THICKNESS,
                ROAD_WIDTH + sidewalkWidth * 2,
              ]}
            />
          </group>
        ))}
        <RoadMarkings />
        <RoadMarkings horizontal={false} />
        {gridNode ? (
          <MarketHub position={gridNode.position} />
        ) : (
          <MarketHub position={[0, 0, 0]} />
        )}
        <MarketHUD
          hubPosition={hubPosition}
          marketSnapshot={marketSnapshot}
          selectedHouseId={selectedHouseId}
          showGridFlows={showGridFlows}
          setShowGridFlows={setShowGridFlows}
          showOrderFlows={showOrderFlows}
          setShowOrderFlows={setShowOrderFlows}
          showTradeFlows={showTradeFlows}
          setShowTradeFlows={setShowTradeFlows}
          maxFlows={maxFlows}
          setMaxFlows={setMaxFlows}
        />
        {households.map((house, index) => {
          const viz = houseStateMap.get(house.id);
          const raw = houseRawStateMap.get(house.id);
          const solarRoof = solarRoofSet.has(house.id);

          const hovered = hoveredHouseId === house.id;
          const selected = selectedHouseId === house.id;

          return (
            <group key={house.id}>
              <PickableHouse
                house={house}
                index={index}
                viz={viz}
                raw={raw}
                daylight={cycle.daylight}
                hasSolarRoof={solarRoof}
                trades={trades}
                hovered={hovered}
                selected={selected}
                onHover={() => setHoveredHouseId(house.id)}
                onUnhover={() =>
                  setHoveredHouseId((prev) => (prev === house.id ? null : prev))
                }
                onToggleSelect={() =>
                  setSelectedHouseId(
                    selectedHouseId === house.id ? null : house.id,
                  )
                }
              />

              {/* keep your decorations as siblings so hover doesn’t flicker */}
              {index % 2 === 0 ? (
                <Tree
                  position={[
                    house.position[0] - 1.14,
                    0,
                    house.position[2] + 0.94,
                  ]}
                />
              ) : null}
              {index % 3 === 0 ? (
                <StreetLamp
                  position={[
                    house.position[0] + 1.62,
                    0,
                    house.position[2] - 0.38,
                  ]}
                />
              ) : null}
              {index % 4 === 0 ? (
                <>
                  <Fence
                    position={[
                      house.position[0] - 0.94,
                      0.02,
                      house.position[2] - 1.02,
                    ]}
                  />
                  <Fence
                    position={[
                      house.position[0] + 0.94,
                      0.02,
                      house.position[2] - 1.02,
                    ]}
                  />
                </>
              ) : null}
            </group>
          );
        })}
        {displayLinks.map((link, index) => {
          const hasSelection = selectedHouseId !== null;
          const involved = hasSelection
            ? linkInvolvesHouse(link, selectedHouseId!)
            : true;

          const fade = !hasSelection ? 1.0 : involved ? 1.0 : 0.1;

          return (
            <EnergyArcFlow
              key={`${link.kind}-${index}`}
              link={link}
              emphasized={hasSelection && involved}
              fade={fade}
            />
          );
        })}
        {/* {focusedLinks.map((link, index) => (
          <EnergyArcFlow
            key={`${link.kind}-${index}`}
            link={link}
            emphasized={!!selectedHouseId}
          />
        ))} */}
        {solarNodes
          .filter((node) => typeof node.serves_household_id !== "number")
          .map((node, index) => (
            <GroundSolarFarm
              key={`solar-${node.id}`}
              position={node.position}
              daylight={cycle.daylight}
              scale={0.95 + (index % 3) * 0.05}
            />
          ))}
        {windNodes.map((node, idx) => (
          <WindTurbine
            key={`wind-${node.id}`}
            position={node.position}
            spinRps={windSpin}
            scale={1.0 + (idx % 2) * 0.08}
          />
        ))}
      </group>
      <ControlsRig
        hoveredHouseId={hoveredHouseId}
        selectedHouseId={selectedHouseId}
        houseById={houseById}
      />
    </>
  );
}

export default function Neighborhood3DScene({
  topology,
  observation,
  latestInfo,
  step,
}: Neighborhood3DSceneProps): JSX.Element {
  const cycle = resolveDayCycle(latestInfo, step);
  const [mounted, setMounted] = useState(false);

  // Only render the Canvas on the client after the component mounts
  useEffect(() => {
    setMounted(true);
    return () => setMounted(false); // Cleanup on unmount/hot-reload
  }, []);

  if (!mounted) return <></>;

  return (
    <Canvas
      camera={{ position: [34, 25, 34], fov: 34 }}
      shadows
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: false, preserveDrawingBuffer: true }}
    >
      <SunRig daylight={cycle.daylight} progress={cycle.progress} />
      <fog attach="fog" args={["#08111f", 85, 220]} />
      <SceneGraph
        topology={topology}
        observation={observation}
        latestInfo={latestInfo}
        step={step}
      />
    </Canvas>
  );
}
