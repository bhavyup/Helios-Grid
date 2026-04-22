"use client";

import dynamic from "next/dynamic";

import { TopologyPayload } from "@/lib/types";

const DynamicNeighborhood3DScene = dynamic(
  () => import("@/components/neighborhood-3d-scene"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-[240px] items-center justify-center text-xs uppercase tracking-[0.16em] text-slate-400">
        Loading 3D Scene...
      </div>
    ),
  },
);

interface Neighborhood3DCardProps {
  topology?: TopologyPayload;
}

export function Neighborhood3DCard({ topology }: Neighborhood3DCardProps): JSX.Element {
  return (
    <section className="panel-surface overflow-hidden px-5 py-5">
      <div className="mb-3">
        <h2 className="panel-title">3D Neighborhood Pulse</h2>
        <p className="mt-1 text-xs text-slate-400">Optional visual layer for faculty demo storytelling and topology clarity.</p>
      </div>
      <div className="overflow-hidden rounded-xl border border-slate-700/70 bg-slate-950/70">
        {topology ? (
          <div className="h-[240px]">
            <DynamicNeighborhood3DScene topology={topology} />
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
