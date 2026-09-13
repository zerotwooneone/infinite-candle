import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useSceneStore } from '../store/sceneStore';

const LED_COUNT = 600;

const LEDMesh = ({ coordsX, coordsY }: { coordsX: number[], coordsY: number[] }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { activeLayers } = useSceneStore();
  
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  
  useEffect(() => {
    if (!meshRef.current) return;
    
    // Geometry translation
    const radius = 21.0 / (2 * Math.PI); // ~3.34
    
    for (let i = 0; i < LED_COUNT; i++) {
      const h = coordsY[i]; // 0.0 to 1.0
      const a = coordsX[i]; // 0.0 to 1.0
      
      const y = (h * 48.0) - 24.0; 
      const theta = a * Math.PI * 2;
      const x = Math.cos(theta) * radius;
      const z = Math.sin(theta) * radius;
      
      dummy.position.set(x, y, z);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
      
      // VERY basic visual proxy: If solid layer exists, use its color. Otherwise fallback.
      // A true proxy would evaluate the generative math here, but we'll do a simple mock.
      let r=0, g=0, b=0;
      
      // Evaluate bottom to top
      activeLayers.forEach(layer => {
        if (h >= (layer.h_min ?? 0) && h <= (layer.h_max ?? 1)) {
          if (layer.type === 'solid' && (layer as any).color) {
            const [lr, lg, lb] = (layer as any).color;
            // blend based on opacity
            const op = layer.opacity ?? 1.0;
            r = r * (1-op) + (lr/255) * op;
            g = g * (1-op) + (lg/255) * op;
            b = b * (1-op) + (lb/255) * op;
          }
          // Mock other layers simply
          if (layer.type === 'fire') {
            const op = layer.opacity ?? 1.0;
            r = r * (1-op) + 1.0 * op;
            g = g * (1-op) + 0.3 * op;
          }
        }
      });
      
      // Default dim ambient if black
      if (r===0 && g===0 && b===0) {
        r = 0.05; g = 0.05; b = 0.05;
      }

      color.setRGB(r, g, b);
      meshRef.current.setColorAt(i, color);
    }
    
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [coordsX, coordsY, activeLayers]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, LED_COUNT]}>
      <sphereGeometry args={[0.2, 8, 8]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export const DigitalTwin: React.FC = () => {
  const [mapping, setMapping] = useState<{coords_x: number[], coords_y: number[]} | null>(null);
  
  useEffect(() => {
    // Attempt to fetch from local hardware
    fetch('http://10.0.217.7:8000/mapping')
      .then(r => r.json())
      .then(data => setMapping(data))
      .catch(e => {
        console.error("Failed to fetch mapping, generating fallback", e);
        // Fallback spiral generation if backend offline
        const cx = [];
        const cy = [];
        const wraps = 19.4;
        for (let i=0; i<LED_COUNT; i++) {
          cy.push(i / LED_COUNT);
          cx.push((i % (LED_COUNT / wraps)) / (LED_COUNT / wraps));
        }
        setMapping({ coords_x: cx, coords_y: cy });
      });
  }, []);

  return (
    <div className="bg-black rounded-lg overflow-hidden h-[600px] border border-zinc-800">
      {mapping ? (
        <Canvas camera={{ position: [0, 0, 40], fov: 50 }}>
          <color attach="background" args={['#050505']} />
          <ambientLight intensity={0.5} />
          <OrbitControls autoRotate autoRotateSpeed={0.5} />
          <LEDMesh coordsX={mapping.coords_x} coordsY={mapping.coords_y} />
        </Canvas>
      ) : (
        <div className="flex items-center justify-center h-full text-zinc-500">
          Loading mapping data...
        </div>
      )}
    </div>
  );
};
