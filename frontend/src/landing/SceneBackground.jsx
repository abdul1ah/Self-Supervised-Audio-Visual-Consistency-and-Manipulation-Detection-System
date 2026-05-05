import { Canvas, useFrame } from "@react-three/fiber";
import { Float } from "@react-three/drei";
import { useRef } from "react";
import * as THREE from "three";

function FloatingForm() {
  const meshRef = useRef(null);

  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.elapsedTime;
    meshRef.current.rotation.x = Math.sin(t * 0.2) * 0.15;
    meshRef.current.rotation.y = t * 0.12;
  });

  return (
    <Float speed={1.4} rotationIntensity={0.25} floatIntensity={0.35}>
      <mesh ref={meshRef} scale={1.15}>
        <icosahedronGeometry args={[1.25, 1]} />
        <meshStandardMaterial
          color="#ebe8e2"
          metalness={0.2}
          roughness={0.35}
          flatShading
        />
      </mesh>
    </Float>
  );
}

export default function SceneBackground() {
  return (
    <div
      aria-hidden
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
      }}
    >
      <Canvas
        camera={{ position: [0, 0.2, 5.5], fov: 40 }}
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: true }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color("#f2f1ec"), 1);
        }}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[5, 6, 4]} intensity={0.85} />
        <directionalLight position={[-4, -2, -3]} intensity={0.35} color="#ffffff" />
        <FloatingForm />
      </Canvas>
    </div>
  );
}
