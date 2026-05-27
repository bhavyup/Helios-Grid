"use client";

import { Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { TopologyPayload } from "@/lib/types";

interface Neighborhood3DSceneProps {
  topology?: TopologyPayload;
}

interface PositionedNode {
  id: number;
  type: string;
  position: [number, number, number];
}

function nodeColor(type: string): string {
  if (type === "grid") {
    return "#2dd4bf";
  }
  if (type === "household") {
    return "#38bdf8";
  }
  if (type === "solar") {
    return "#fbbf24";
  }
  return "#86efac";
}

function SceneGraph({ topology }: Neighborhood3DSceneProps): JSX.Element {
  const groupRef = useRef<THREE.Group>(null);

  const positionedNodes = useMemo<PositionedNode[]>(() => {
    const nodes = topology?.nodes ?? [];
    const gridNode = nodes.find((node) => node.type === "grid");
    const households = nodes.filter((node) => node.type === "household");
    const producers = nodes.filter((node) => node.type !== "grid" && node.type !== "household");

    const resolved: PositionedNode[] = [];

    if (gridNode) {
      resolved.push({ id: gridNode.id, type: gridNode.type, position: [0, 0, 0] });
    }

    households.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(households.length, 1);
      resolved.push({
        id: node.id,
        type: node.type,
        position: [Math.cos(angle) * 2.2, Math.sin(angle) * 0.6, Math.sin(angle) * 2.2],
      });
    });

    producers.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(producers.length, 1);
      resolved.push({
        id: node.id,
        type: node.type,
        position: [Math.cos(angle) * 3.2, 0.8, Math.sin(angle) * 3.2],
      });
    });

    return resolved;
  }, [topology]);

  const positionLookup = useMemo(() => {
    const map = new Map<number, [number, number, number]>();
    positionedNodes.forEach((node) => map.set(node.id, node.position));
    return map;
  }, [positionedNodes]);

  useFrame((_, delta) => {
    if (!groupRef.current) {
      return;
    }
    groupRef.current.rotation.y += delta * 0.09;
  });

  return (
    <group ref={groupRef}>
      {(topology?.edges ?? []).slice(0, 64).map((edge) => {
        const source = positionLookup.get(edge.source);
        const target = positionLookup.get(edge.target);
        if (!source || !target) {
          return null;
        }

        return (
          <Line
            key={`${edge.source}-${edge.target}`}
            points={[source, target]}
            color="#38bdf8"
            transparent
            opacity={0.35}
            lineWidth={1.1}
          />
        );
      })}

      {positionedNodes.map((node) => (
        <mesh key={node.id} position={node.position}>
          <sphereGeometry args={[node.type === "grid" ? 0.18 : 0.11, 24, 24]} />
          <meshStandardMaterial color={nodeColor(node.type)} emissive={nodeColor(node.type)} emissiveIntensity={0.25} />
        </mesh>
      ))}
    </group>
  );
}

export default function Neighborhood3DScene({ topology }: Neighborhood3DSceneProps): JSX.Element {
  return (
    <Canvas camera={{ position: [0, 4.2, 7.4], fov: 42 }}>
      <color attach="background" args={["#020617"]} />
      <ambientLight intensity={0.65} />
      <pointLight position={[3, 4, 4]} intensity={1.15} color="#67e8f9" />
      <pointLight position={[-4, 2, -3]} intensity={0.65} color="#fbbf24" />
      <SceneGraph topology={topology} />
      <OrbitControls enablePan={false} minDistance={5.5} maxDistance={10} autoRotate autoRotateSpeed={0.35} />
    </Canvas>
  );
}
