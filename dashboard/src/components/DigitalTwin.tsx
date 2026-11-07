import React, { useEffect, useState, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useSceneStore } from '../store/sceneStore';
import { API_BASE, WS_STREAM_URL } from '../config';
import { simulateFrame } from '../sim/engineSim';
import { Maximize2, Minimize2, RotateCcw, Play, Pause } from 'lucide-react';

const LED_COUNT = 600;
const FRAME_BYTES = LED_COUNT * 3;

// Camera fit: pillar is 48 units tall, radius ~3.34. On mount we solve the
// camera distance from the pillar's bounding sphere against the SMALLER of
// the vertical/horizontal FOV, so the whole structure is framed on any aspect.
const PILLAR_HEIGHT = 48;
const PILLAR_RADIUS = 21.0 / (2 * Math.PI);
const CAM_FOV = 50;

const CameraRig = ({ resetTrigger }: { resetTrigger: number }) => {
  const camera = useThree(s => s.camera) as THREE.PerspectiveCamera;
  const size = useThree(s => s.size);

  useEffect(() => {
    const aspect = size.width / size.height;
    if (!Number.isFinite(aspect) || aspect <= 0) return;

    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
    const fitFov = Math.min(vFov, hFov);

    const boundingRadius = Math.hypot(PILLAR_HEIGHT / 2, PILLAR_RADIUS);
    const dist = (boundingRadius / Math.sin(fitFov / 2)) * 1.15; // 15% margin

    camera.position.set(0, 4, dist);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
  }, [resetTrigger, size.width, size.height, camera]);

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
  }, [coordsX, coordsY, dummy]);

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
  }, [coordsY, activeLayers, color, frameRef]);

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
};

export const DigitalTwin: React.FC<{ className?: string }> = ({ className = '' }) => {
  const [mapping, setMapping] = useState<{ coords_x: number[], coords_y: number[] } | null>(null);
  const [live, setLive] = useState(false);
  const [devMode, setDevMode] = useState(false);
  const [viewMode, setViewMode] = useState<'xray' | 'realistic'>('realistic');
  const [autoRotate, setAutoRotate] = useState(true);
  const [resetTrigger, setResetTrigger] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const frameRef = useRef<Uint8Array | null>(null);

  // Handle escape key to exit fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullscreen]);

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
    <div
      className={`bg-zinc-950 rounded-xl overflow-hidden border border-zinc-800 relative transition-all duration-300 flex flex-col ${
        isFullscreen
          ? 'fixed inset-0 z-50 rounded-none border-0 h-screen w-screen'
          : `h-full min-h-[300px] sm:min-h-[400px] md:min-h-[480px] lg:min-h-[520px] ${className}`
      }`}
    >
      {/* Top Controls Overlay */}
      <div className="absolute top-0 left-0 right-0 z-10 p-2 sm:p-3 flex items-center justify-between gap-2 pointer-events-none bg-gradient-to-b from-zinc-950/80 via-zinc-950/40 to-transparent">
        {/* Status Pill */}
        <div className="pointer-events-auto flex items-center gap-1.5">
          <span className="text-[11px] sm:text-xs px-2.5 py-1 rounded-full bg-zinc-900/90 border border-zinc-700/80 backdrop-blur-md shadow-sm flex items-center gap-1.5 font-medium">
            {devMode ? (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-amber-300">DEV SIM</span>
              </>
            ) : live ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-emerald-300">LIVE</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-zinc-500" />
                <span className="text-zinc-400">OFFLINE</span>
              </>
            )}
          </span>
        </div>

        {/* Action Controls */}
        <div className="pointer-events-auto flex items-center flex-wrap justify-end gap-1 sm:gap-1.5">
          {/* Dev Mode toggle */}
          <button
            onClick={() => setDevMode(d => !d)}
            className={`text-[11px] sm:text-xs px-2.5 py-1 rounded-lg border font-medium backdrop-blur-md transition-all active:scale-95 shadow-sm ${
              devMode
                ? 'bg-amber-950/70 border-amber-600/80 text-amber-200 hover:bg-amber-900/80'
                : 'bg-zinc-900/90 border-zinc-700/80 text-zinc-300 hover:text-white hover:bg-zinc-800'
            }`}
          >
            {devMode ? 'Live Mode' : 'Dev Sim'}
          </button>

          {/* View Mode Switcher */}
          <div className="flex rounded-lg border border-zinc-700/80 bg-zinc-900/90 backdrop-blur-md overflow-hidden text-[11px] sm:text-xs shadow-sm">
            <button
              onClick={() => setViewMode('xray')}
              className={`px-2 py-1 font-medium transition-colors ${
                viewMode === 'xray' ? 'bg-zinc-700 text-white font-semibold' : 'text-zinc-400 hover:text-white'
              }`}
            >
              X-Ray
            </button>
            <button
              onClick={() => setViewMode('realistic')}
              className={`px-2 py-1 font-medium transition-colors ${
                viewMode === 'realistic' ? 'bg-zinc-700 text-white font-semibold' : 'text-zinc-400 hover:text-white'
              }`}
            >
              Realistic
            </button>
          </div>

          {/* Auto-Rotate Toggle */}
          <button
            onClick={() => setAutoRotate(r => !r)}
            title={autoRotate ? 'Pause Rotation' : 'Auto Rotate'}
            className={`p-1.5 rounded-lg border backdrop-blur-md transition-all active:scale-95 shadow-sm ${
              autoRotate
                ? 'bg-blue-600/30 border-blue-500/60 text-blue-300'
                : 'bg-zinc-900/90 border-zinc-700/80 text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            {autoRotate ? <Pause size={14} /> : <Play size={14} />}
          </button>

          {/* Reset Camera View */}
          <button
            onClick={() => setResetTrigger(t => t + 1)}
            title="Reset View"
            className="p-1.5 rounded-lg border border-zinc-700/80 bg-zinc-900/90 backdrop-blur-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all active:scale-95 shadow-sm"
          >
            <RotateCcw size={14} />
          </button>

          {/* Fullscreen / Expand Toggle */}
          <button
            onClick={() => setIsFullscreen(f => !f)}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen 3D View'}
            className={`p-1.5 rounded-lg border backdrop-blur-md transition-all active:scale-95 shadow-sm ${
              isFullscreen
                ? 'bg-amber-500/20 border-amber-500/60 text-amber-300'
                : 'bg-zinc-900/90 border-zinc-700/80 text-zinc-400 hover:text-white hover:bg-zinc-800'
            }`}
          >
            {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      </div>

      {/* 3D Canvas or Loading */}
      <div className="flex-1 w-full h-full min-h-0 relative touch-none">
        {mapping ? (
          <Canvas camera={{ position: [0, 4, 62], fov: CAM_FOV }}>
            <color attach="background" args={['#050505']} />
            <ambientLight intensity={0.5} />
            <CameraRig resetTrigger={resetTrigger} />
            <OrbitControls
              autoRotate={autoRotate}
              autoRotateSpeed={0.6}
              enableDamping={true}
              dampingFactor={0.05}
              makeDefault
            />
            <LEDMesh
              coordsX={mapping.coords_x}
              coordsY={mapping.coords_y}
              frameRef={frameRef}
              devMode={devMode}
              realistic={viewMode === 'realistic'}
            />
          </Canvas>
        ) : (
          <div className="flex flex-col items-center justify-center h-full min-h-[260px] text-zinc-500 gap-2">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-xs">Loading physical LED mapping...</span>
          </div>
        )}
      </div>

      {/* Touch drag helper prompt (shown subtle on mobile) */}
      <div className="absolute bottom-2 left-3 pointer-events-none text-[10px] text-zinc-500 bg-zinc-950/60 px-2 py-0.5 rounded backdrop-blur-sm sm:hidden">
        Drag to rotate • Pinch to zoom
      </div>
    </div>
  );
};
