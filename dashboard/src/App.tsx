import React, { useEffect, useState } from 'react';
import { SceneDashboard } from './components/SceneDashboard';
import { LayerManager } from './components/LayerManager';
import { LayerInspector } from './components/LayerInspector';
import { ActionBar } from './components/ActionBar';
import { DigitalTwin } from './components/DigitalTwin';
import { useSceneStore, FACTORY_PRESETS } from './store/sceneStore';

function App() {
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const { savedScenes, saveScene, activeLayers } = useSceneStore();
  
  // Initialize with empty factory preset behavior
  useEffect(() => {
    if (savedScenes.length === 0 && activeLayers.length === 0) {
      // It's pristine, let's just leave it empty or load the first factory preset
    }
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
