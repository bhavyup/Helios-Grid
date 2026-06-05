"use client";

import dynamic from "next/dynamic";

import { SimulationObservation, TopologyPayload } from "@/lib/types";

const DynamicNeighborhood3DScene = dynamic(
  () => import("./neighborhood-3d-scene"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[240px] items-center justify-center text-xs uppercase tracking-[0.16em] text-slate-400">
        Loading city block...
      </div>
    ),
  },
);

interface Neighborhood3DCardProps {
  topology?: TopologyPayload;
  observation?: SimulationObservation;
  latestInfo?: Record<string, unknown>;
  step?: number;
}

export function Neighborhood3DCard({ topology, observation, latestInfo, step }: Neighborhood3DCardProps): JSX.Element {
  return (
    <section className="panel-surface flex flex-col overflow-hidden px-5 py-5 lg:px-6 lg:py-6">
      <div className="mb-4">
        <p className="section-eyebrow">Grid simulation</p>
        <h2 className="hero-display mt-2 text-3xl font-semibold tracking-[-0.03em] text-white">Living 3D locality</h2>
        <p className="section-copy mt-2 text-sm">Daylight, nightfall, solar generation, demand draw, and P2P energy flow animate across the neighborhood.</p>
      </div>
      <div className="overflow-hidden h-[40rem] rounded-[1.35rem] border border-white/10 bg-[rgba(255,255,255,0.02)]">
        {topology ? (
          <div className="h-full">
            <DynamicNeighborhood3DScene topology={topology} observation={observation} latestInfo={latestInfo} step={step} />
          </div>
        ) : (
          <div className="flex h-[240px] items-center justify-center text-xs uppercase tracking-[0.16em] text-slate-400">
            Reset simulation to load topology.
          </div>
        )}
      </div>
    </section>
  );
}
