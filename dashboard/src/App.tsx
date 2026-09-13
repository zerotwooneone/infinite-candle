import React, { useEffect, useState } from 'react';
import { SceneDashboard } from './components/SceneDashboard';
import { LayerManager } from './components/LayerManager';
import { LayerInspector } from './components/LayerInspector';
import { ActionBar } from './components/ActionBar';
import { DigitalTwin } from './components/DigitalTwin';
import { useSceneStore } from './store/sceneStore';
import { API_BASE } from './config';

const generateId = () => Math.random().toString(36).substring(2, 9);

function App() {
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const { setActiveLayers } = useSceneStore();

  // Rehydrate from the Pi's authoritative state on mount.
  // The backend is the source of truth — localStorage is just a draft cache.
  useEffect(() => {
    fetch(`${API_BASE}/scene`)
      .then(r => r.json())
      .then(data => {
        if (data?.layers?.length) {
          setActiveLayers(data.layers.map((l: any) => ({ ...l, id: generateId() })));
        }
      })
      .catch(e => console.warn('Could not fetch active scene from Pi:', e));
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 pb-24">
      <header className="bg-zinc-900 border-b border-zinc-800 p-4">
        <h1 className="text-2xl font-bold text-white tracking-wider flex items-center">
          <span className="mr-2">🕯️</span> Infinite Candle Control
        </h1>
      </header>

      <main className="p-6 max-w-7xl mx-auto grid grid-cols-12 gap-6">
        {/* Left Column: Presets & Inspector */}
        <div className="col-span-12 md:col-span-3 space-y-6">
          <SceneDashboard />
          <LayerInspector selectedLayerId={selectedLayerId} />
        </div>
        
        {/* Middle Column: Layer Stack */}
        <div className="col-span-12 md:col-span-4">
          <LayerManager selectedLayerId={selectedLayerId} setSelectedLayerId={setSelectedLayerId} />
        </div>

        {/* Right Column: 3D Twin */}
        <div className="col-span-12 md:col-span-5">
          <DigitalTwin />
        </div>
      </main>

      <ActionBar />
    </div>
  );
}

export default App;
