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
    <div className="h-screen bg-zinc-950 flex flex-col overflow-hidden">
      <header className="bg-zinc-900 border-b border-zinc-800 p-4 shrink-0">
        <h1 className="text-2xl font-bold text-white tracking-wider flex items-center">
          <span className="mr-2">🕯️</span> Infinite Candle Control
        </h1>
      </header>

      <main className="flex-1 min-h-0 w-full max-w-[1920px] mx-auto p-6 pb-24 grid grid-cols-12 gap-6">
        {/* Left half: controls (presets + inspector | layer stack) */}
        <div className="col-span-12 min-[1700px]:col-span-6 min-h-0 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-6 content-start">
          <div className="space-y-6">
            <SceneDashboard />
            <LayerInspector selectedLayerId={selectedLayerId} />
          </div>
          <LayerManager selectedLayerId={selectedLayerId} setSelectedLayerId={setSelectedLayerId} />
        </div>

        {/* Right half: 3D twin — ~50% of screen, min 800x600 */}
        <div className="col-span-12 min-[1700px]:col-span-6 min-h-[600px]">
          <DigitalTwin />
        </div>
      </main>

      <ActionBar />
    </div>
  );
}

export default App;
