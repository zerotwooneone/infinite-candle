import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useSceneStore } from '../store/sceneStore';
import { API_BASE, WS_STREAM_URL } from '../config';
import { simulateFrame } from '../sim/engineSim';

const LED_COUNT = 600;
const FRAME_BYTES = LED_COUNT * 3;

// Camera fit: pillar is 48 units tall, radius ~3.34. On mount we solve the
// camera distance from the pillar's bounding sphere against the SMALLER of
// the vertical/horizontal FOV, so the whole structure is framed on any aspect.
const PILLAR_HEIGHT = 48;
const PILLAR_RADIUS = 21.0 / (2 * Math.PI);
const CAM_FOV = 50;

const CameraRig = () => {
  const camera = useThree(s => s.camera) as THREE.PerspectiveCamera;
  const size = useThree(s => s.size);

  useEffect(() => {
    const aspect = size.width / size.height;
    if (!Number.isFinite(aspect) || aspect <= 0) return;

    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
    const fitFov = Math.min(vFov, hFov);

    const boundingRadius = Math.hypot(PILLAR_HEIGHT / 2, PILLAR_RADIUS);
    const dist = (boundingRadius / Math.sin(fitFov / 2)) * 1.1; // 10% margin

    camera.position.set(0, 4, dist);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // fit once on mount — never fight the user's zoom/orbit

  return null;
};

type FrameRef = React.MutableRefObject<Uint8Array | null>;

// Material with a patched shader: each instance carries an outward-facing
// normal (aOutward). When uRealistic=1, fragments viewed from behind the
// strip face are dimmed to ~3% — the LED's backing, not its emitter.
const createLedMaterial = () => {
  const m = new THREE.MeshBasicMaterial({ toneMapped: false });
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uRealistic = { value: 0 };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>',
        '#include <common>\nattribute vec3 aOutward;\nvarying vec3 vOutward;\nvarying vec3 vWorldPos;')
      .replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvOutward = aOutward;\nvWorldPos = (modelMatrix * instanceMatrix * vec4(position, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>',
        '#include <common>\nuniform float uRealistic;\nvarying vec3 vOutward;\nvarying vec3 vWorldPos;')
      .replace('#include <color_fragment>',
        `#include <color_fragment>
        if (uRealistic > 0.5) {
          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          float facing = smoothstep(-0.15, 0.25, dot(normalize(vOutward), viewDir));
          diffuseColor.rgb *= mix(0.03, 1.0, facing);
        }`);
    m.userData.shader = shader;
  };
  return m;
};

const LEDMesh = ({ coordsX, coordsY, frameRef, devMode, realistic }: { coordsX: number[], coordsY: number[], frameRef: FrameRef, devMode: boolean, realistic: boolean }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { activeLayers } = useSceneStore();

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);
  const simBuf = useMemo(() => new Uint8Array(FRAME_BYTES), []);
  const ledMaterial = useMemo(createLedMaterial, []);

  // Position instances once — geometry never changes.
  useEffect(() => {
    if (!meshRef.current) return;

    const radius = 21.0 / (2 * Math.PI); // ~3.34

    for (let i = 0; i < LED_COUNT; i++) {
      const h = coordsY[i];
      const a = coordsX[i];

      const y = (h * 48.0) - 24.0;
      const theta = a * Math.PI * 2;
      dummy.position.set(Math.cos(theta) * radius, y, Math.sin(theta) * radius);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;

    // Per-instance outward normal (radial direction) for backside dimming.
    const outward = new Float32Array(LED_COUNT * 3);
    for (let i = 0; i < LED_COUNT; i++) {
      const theta = coordsX[i] * Math.PI * 2;
      outward[i * 3] = Math.cos(theta);
      outward[i * 3 + 1] = 0;
      outward[i * 3 + 2] = Math.sin(theta);
    }
    meshRef.current.geometry.setAttribute('aOutward', new THREE.InstancedBufferAttribute(outward, 3));
  }, [coordsX, coordsY]);

  // Flip the shader uniform when the view mode toggles (no recompile needed).
  useEffect(() => {
    const shader = ledMaterial.userData.shader;
    if (shader) shader.uniforms.uRealistic.value = realistic ? 1 : 0;
  }, [realistic, ledMaterial]);

  // Static proxy colors — only applied when no live stream is available.
  useEffect(() => {
    if (!meshRef.current || frameRef.current) return;

    for (let i = 0; i < LED_COUNT; i++) {
      const h = coordsY[i];
      let r = 0, g = 0, b = 0;

      activeLayers.forEach(layer => {
        if (h >= (layer.h_min ?? 0) && h <= (layer.h_max ?? 1)) {
          const op = layer.opacity ?? 1.0;
          if (layer.type === 'solid' && (layer as any).color) {
            const [lr, lg, lb] = (layer as any).color;
            r = r * (1 - op) + (lr / 255) * op;
            g = g * (1 - op) + (lg / 255) * op;
            b = b * (1 - op) + (lb / 255) * op;
          }
          if (layer.type === 'fire') {
            r = r * (1 - op) + 1.0 * op;
            g = g * (1 - op) + 0.3 * op;
          }
        }
      });

      if (r === 0 && g === 0 && b === 0) { r = 0.05; g = 0.05; b = 0.05; }

      color.setRGB(r, g, b);
      meshRef.current.setColorAt(i, color);
    }
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [coordsY, activeLayers]);

  // Per-tick color update: live stream wins, then dev-mode sim, else keep static proxy.
  useFrame(({ clock }) => {
    if (!meshRef.current) return;

    let frame = frameRef.current;
    if (devMode) {
      simulateFrame(activeLayers, clock.elapsedTime, coordsX, coordsY, simBuf);
      frame = simBuf;
    }
    if (!frame || frame.length < FRAME_BYTES) return;

    for (let i = 0; i < LED_COUNT; i++) {
      const o = i * 3;
      color.setRGB(frame[o] / 255, frame[o + 1] / 255, frame[o + 2] / 255);
      meshRef.current.setColorAt(i, color);
    }
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  });

  return (
    <group>
      {realistic && (
        // Opaque pillar body — occludes the far side of the strip.
        <mesh>
          <cylinderGeometry args={[PILLAR_RADIUS - 0.3, PILLAR_RADIUS - 0.3, PILLAR_HEIGHT, 32]} />
          <meshBasicMaterial color="#0b0b0d" />
        </mesh>
      )}
      <instancedMesh ref={meshRef} args={[undefined, undefined, LED_COUNT]}>
        <sphereGeometry args={[0.2, 8, 8]} />
        <primitive object={ledMaterial} attach="material" />
      </instancedMesh>
    </group>
  );
}

export const DigitalTwin: React.FC = () => {
  const [mapping, setMapping] = useState<{ coords_x: number[], coords_y: number[] } | null>(null);
  const [live, setLive] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [viewMode, setViewMode] = useState<'xray' | 'realistic'>('realistic');
  const frameRef = useRef<Uint8Array | null>(null);

  // Fetch physical mapping (with offline fallback).
  useEffect(() => {
    fetch(`${API_BASE}/mapping`)
      .then(r => r.json())
      .then(data => setMapping(data))
      .catch(e => {
        console.error("Failed to fetch mapping, generating fallback", e);
        const cx: number[] = [];
        const cy: number[] = [];
        const wraps = 19.4;
        for (let i = 0; i < LED_COUNT; i++) {
          cy.push(i / LED_COUNT);
          cx.push((i % (LED_COUNT / wraps)) / (LED_COUNT / wraps));
        }
        setMapping({ coords_x: cx, coords_y: cy });
      });
  }, []);

  // Live frame stream from the Pi — raw 1800-byte binary frames at ~15 FPS.
  // Skipped entirely in dev mode so the dashboard works without hardware.
  useEffect(() => {
    if (devMode) {
      setLive(false);
      frameRef.current = null;
      return;
    }

    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      ws = new WebSocket(WS_STREAM_URL);
      ws.binaryType = 'arraybuffer';

      ws.onmessage = (evt) => {
        frameRef.current = new Uint8Array(evt.data as ArrayBuffer);
      };
      ws.onopen = () => setLive(true);
      ws.onclose = () => {
        setLive(false);
        frameRef.current = null;
        if (!disposed) retry = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws?.close();
    };

    connect();
    return () => {
      disposed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [devMode]);

  return (
    <div className="bg-black rounded-lg overflow-hidden h-full min-h-[600px] border border-zinc-800 relative">
      <div className="absolute top-2 right-2 z-10 flex items-center gap-2">
        <span className="text-xs px-2 py-1 rounded bg-zinc-900/80 border border-zinc-700">
          {devMode
            ? <span className="text-amber-400">● DEV SIM</span>
            : live
              ? <span className="text-green-400">● LIVE</span>
              : <span className="text-zinc-500">○ OFFLINE (proxy)</span>}
        </span>
        <button
          onClick={() => setDevMode(d => !d)}
          className={`text-xs px-2 py-1 rounded border transition-colors ${
            devMode
              ? 'bg-amber-900/60 border-amber-700 text-amber-300'
              : 'bg-zinc-900/80 border-zinc-700 text-zinc-400 hover:text-white'
          }`}
        >
          {devMode ? 'Go Live' : 'Dev Mode'}
        </button>
        <div className="flex rounded border border-zinc-700 overflow-hidden text-xs">
          <button
            onClick={() => setViewMode('xray')}
            className={`px-2 py-1 transition-colors ${
              viewMode === 'xray' ? 'bg-zinc-700 text-white' : 'bg-zinc-900/80 text-zinc-400 hover:text-white'
            }`}
          >
            X-Ray
          </button>
          <button
            onClick={() => setViewMode('realistic')}
            className={`px-2 py-1 transition-colors ${
              viewMode === 'realistic' ? 'bg-zinc-700 text-white' : 'bg-zinc-900/80 text-zinc-400 hover:text-white'
            }`}
          >
            Realistic
          </button>
        </div>
      </div>
      {mapping ? (
        <Canvas camera={{ position: [0, 4, 62], fov: CAM_FOV }}>
          <color attach="background" args={['#050505']} />
          <ambientLight intensity={0.5} />
          <CameraRig />
          <OrbitControls autoRotate autoRotateSpeed={0.5} makeDefault />
          <LEDMesh coordsX={mapping.coords_x} coordsY={mapping.coords_y} frameRef={frameRef} devMode={devMode} realistic={viewMode === 'realistic'} />
        </Canvas>
      ) : (
        <div className="flex items-center justify-center h-full text-zinc-500">
          Loading mapping data...
        </div>
      )}
    </div>
  );
};
