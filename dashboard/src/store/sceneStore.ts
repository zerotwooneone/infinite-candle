import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { LayerConfig } from '../types/schemas';

const generateId = () => Math.random().toString(36).substring(2, 9);

export const FACTORY_PRESETS = [
  {
    name: 'Alien',
    layers: [
      { id: generateId(), type: 'solid', color: [0, 0, 1] },
      { id: generateId(), type: 'alien', speed: 1.0, transparent: true, ship_color_1: [50, 150, 255], ship_color_2: [0, 255, 50], beam_color: [100, 255, 255] }
    ]
  },
  {
    name: 'Fireworks 1',
    layers: [
      {
        id: generateId(),
        type: 'fireworks', opacity: 1.0, h_min: 0.0, h_max: 1.0, launch_rate: 1.2, burst_height: 0.78, explosion_size: 0.02,
        max_rockets: 2, max_sparks: 400, spark_density: 0.8, rocket_speed: 0.99, rocket_wiggle: 0.02, rocket_gravity: -0.35,
        spark_gravity: -0.85, spark_drag: 0.50, trail_decay: 2.8, brightness: 1.0
      }
    ]
  },
  {
    name: 'Candy Cane',
    layers: [
      { id: generateId(), type: 'stripes', color_a: [55, 0, 0], color_b: [0, 55, 0], angle: 45.0, width: 0.5, speed: 1 },
      { id: generateId(), type: 'clip', filename: 'snow_heavy.npy', transparent: true }
    ]
  },
  {
    name: 'Candle',
    layers: [
      { id: generateId(), type: 'solid', color: [50, 45, 40], h_min: 0.0, h_max: 0.55 },
      { id: generateId(), type: 'fire', h_min: 0.55, h_max: 1.0, color_start: [255, 180, 40], color_end: [255, 0, 0], cooling: 3.2, sparking: 0.5 }
    ]
  },
  {
    name: 'Snow',
    layers: [
      { id: generateId(), type: 'solid', color: [1, 0, 2] },
      { id: generateId(), type: 'clip', filename: 'snow_heavy.npy', transparent: true }
    ]
  },
  {
    name: 'Lava Lamp',
    layers: [
      { id: generateId(), type: 'lava', color: [255, 0, 0], bg_color: [0, 200, 0], blob_count: 6, speed: 0.8, opacity: 1.0 }
    ]
  },
  {
    name: 'Geometric Wireframes',
    layers: [
      { id: generateId(), type: 'solid', color: [10, 0, 20] },
      { id: generateId(), type: 'clip', filename: 'wireframes_v1.npy', transparent: true }
    ]
  },
  {
    name: 'Plasma',
    layers: [
      { id: generateId(), type: 'solid', color: [0, 0, 30] },
      { id: generateId(), type: 'clip', filename: 'plasma_rainbow.npy', transparent: true, opacity: 1.0 }
    ]
  },
  {
    name: 'Chrome',
    layers: [
      { id: generateId(), type: 'clip', filename: 'chrome_spin.npy', transparent: false }
    ]
  },
  {
    name: "Rubik's Cube",
    layers: [
      { id: generateId(), type: 'clip', filename: 'rubiks_solve_loop.npy', transparent: true }
    ]
  },
  {
    name: 'Lighthouse',
    layers: [
      // Tower body: dim red/white bands. angle≈24° is the shallowest helix that
      // yields ~5 band pairs (stripes can't do true horizontal — freq_x >= 1).
      { id: generateId(), type: 'stripes', color_a: [110, 0, 0], color_b: [95, 95, 95], angle: 24.0, width: 0.1, speed: 0, h_min: 0.0, h_max: 0.85 },
      // Lantern room: two opposite ~90° white wedges sweeping at ~5s/rev.
      { id: generateId(), type: 'stripes', color_a: [255, 255, 255], color_b: [0, 0, 0], angle: 0.0, width: 0.25, speed: 1.3, h_min: 0.85, h_max: 1.0 }
    ]
  }
] as any[];

interface SceneState {
  activeLayers: LayerConfig[];
  savedScenes: { name: string; layers: LayerConfig[] }[];
  setActiveLayers: (layers: LayerConfig[]) => void;
  updateLayer: (id: string, updates: Partial<LayerConfig>) => void;
  removeLayer: (id: string) => void;
  addLayer: (layer: LayerConfig) => void;
  moveLayer: (id: string, direction: 'up' | 'down') => void;
  saveScene: (name: string) => void;
}

export const useSceneStore = create<SceneState>()(
  persist(
    (set) => ({
      activeLayers: [],
      savedScenes: [],
      
      setActiveLayers: (layers) => set({ activeLayers: layers }),
      
      updateLayer: (id, updates) => set((state) => ({
        activeLayers: state.activeLayers.map((l) => (l.id === id ? { ...l, ...updates } : l)) as LayerConfig[]
      })),
      
      removeLayer: (id) => set((state) => ({
        activeLayers: state.activeLayers.filter((l) => l.id !== id)
      })),
      
      addLayer: (layer) => set((state) => ({
        activeLayers: [...state.activeLayers, layer]
      })),
      
      moveLayer: (id, direction) => set((state) => {
        const index = state.activeLayers.findIndex((l) => l.id === id);
        if (index < 0) return state;
        if (direction === 'up' && index === state.activeLayers.length - 1) return state;
        if (direction === 'down' && index === 0) return state;
        
        const newLayers = [...state.activeLayers];
        const swapIndex = direction === 'up' ? index + 1 : index - 1;
        
        [newLayers[index], newLayers[swapIndex]] = [newLayers[swapIndex], newLayers[index]];
        
        return { activeLayers: newLayers };
      }),
      
      saveScene: (name) => set((state) => ({
        savedScenes: [...state.savedScenes, { name, layers: state.activeLayers }]
      }))
    }),
    {
      name: 'infinite-candle-storage',
    }
  )
);
