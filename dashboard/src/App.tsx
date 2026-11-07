import { useEffect, useState } from 'react';
import { SceneDashboard } from './components/SceneDashboard';
import { LayerManager } from './components/LayerManager';
import { LayerInspector } from './components/LayerInspector';
import { ActionBar } from './components/ActionBar';
import { DigitalTwin } from './components/DigitalTwin';
import { useSceneStore } from './store/sceneStore';
import { API_BASE } from './config';
import { Eye, Layers, Sparkles, LayoutGrid } from 'lucide-react';

const generateId = () => Math.random().toString(36).substring(2, 9);

type MobileTab = 'all' | 'twin' | 'layers' | 'presets';

function App() {
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<MobileTab>('all');
  const { setActiveLayers, setPlugins } = useSceneStore();

  // Rehydrate from the Pi's authoritative state on mount.
  // The backend is the source of truth — localStorage is just a draft cache.
  useEffect(() => {
    fetch(`${API_BASE}/scene`)
      .then(r => r.json())
      .then(data => {
        if (data?.layers?.length) {
          setActiveLayers(data.layers.map((l: any) => ({ ...l, id: l.id || generateId() })));
        }
      })
      .catch(e => console.warn('Could not fetch active scene from Pi:', e));
  }, [setActiveLayers]);

  // Discover registered plugins — drives the "+ Add Layer" menu and the
  // plugin-owned inspector controls.
  useEffect(() => {
    fetch(`${API_BASE}/plugins`)
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data?.plugins)) setPlugins(data.plugins);
      })
      .catch(e => console.warn('Could not fetch plugin metadata from Pi:', e));
  }, [setPlugins]);

  return (
    <div className="min-h-screen min-h-[100dvh] bg-zinc-950 flex flex-col text-slate-200">
      {/* Header */}
      <header className="bg-zinc-900/90 backdrop-blur-md border-b border-zinc-800 px-4 sm:px-6 py-3 shrink-0 sticky top-0 z-30 pt-[calc(0.75rem+env(safe-area-inset-top,0px))]">
        <div className="max-w-[1920px] mx-auto flex items-center justify-between">
          <h1 className="text-lg sm:text-xl md:text-2xl font-bold text-white tracking-wider flex items-center gap-2">
            <span className="text-xl sm:text-2xl animate-pulse">🕯️</span>
            <span>Infinite Candle Control</span>
          </h1>

          <div className="hidden sm:flex items-center gap-2 text-xs text-zinc-400 bg-zinc-950/60 px-3 py-1.5 rounded-full border border-zinc-800">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="font-mono text-zinc-300">Ready</span>
          </div>
        </div>

        {/* Mobile / Tablet View Switcher (< lg) */}
        <div className="lg:hidden mt-3 flex items-center gap-1 bg-zinc-950/80 p-1 rounded-xl border border-zinc-800/80 overflow-x-auto">
          <button
            onClick={() => setActiveTab('all')}
            className={`flex-1 py-1.5 px-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all whitespace-nowrap active:scale-95 ${
              activeTab === 'all'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
            }`}
          >
            <LayoutGrid size={13} />
            <span>Overview</span>
          </button>
          <button
            onClick={() => setActiveTab('twin')}
            className={`flex-1 py-1.5 px-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all whitespace-nowrap active:scale-95 ${
              activeTab === 'twin'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
            }`}
          >
            <Eye size={13} />
            <span>3D View</span>
          </button>
          <button
            onClick={() => setActiveTab('layers')}
            className={`flex-1 py-1.5 px-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all whitespace-nowrap active:scale-95 ${
              activeTab === 'layers'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
            }`}
          >
            <Layers size={13} />
            <span>Layers</span>
          </button>
          <button
            onClick={() => setActiveTab('presets')}
            className={`flex-1 py-1.5 px-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all whitespace-nowrap active:scale-95 ${
              activeTab === 'presets'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
            }`}
          >
            <Sparkles size={13} />
            <span>Presets</span>
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 min-h-0 w-full max-w-[1920px] mx-auto p-3 sm:p-4 md:p-6 pb-28 sm:pb-32">
        {/* DESKTOP LAYOUT (>= 1024px lg): Side-by-side responsive split */}
        <div className="hidden lg:grid lg:grid-cols-12 gap-6 h-[calc(100dvh-140px)] min-h-[550px]">
          {/* Controls Column (Left): 6 or 7 cols */}
          <div className="col-span-6 xl:col-span-7 h-full overflow-y-auto pr-2 grid grid-cols-1 xl:grid-cols-2 gap-6 content-start">
            <div className="space-y-6">
              <SceneDashboard />
              <LayerInspector selectedLayerId={selectedLayerId} />
            </div>
            <div className="h-full">
              <LayerManager selectedLayerId={selectedLayerId} setSelectedLayerId={setSelectedLayerId} />
            </div>
          </div>

          {/* 3D Digital Twin Column (Right): 6 or 5 cols */}
          <div className="col-span-6 xl:col-span-5 h-full">
            <DigitalTwin className="h-full" />
          </div>
        </div>

        {/* MOBILE & TABLET LAYOUT (< 1024px) */}
        <div className="lg:hidden space-y-4">
          {/* Tab: Overview (All) */}
          {activeTab === 'all' && (
            <div className="space-y-4">
              {/* Compact 3D preview card on top */}
              <div className="w-full h-[280px] sm:h-[360px] md:h-[420px]">
                <DigitalTwin className="h-full" />
              </div>

              {/* Layer Stack & Inspector */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <LayerManager selectedLayerId={selectedLayerId} setSelectedLayerId={setSelectedLayerId} />
                <LayerInspector selectedLayerId={selectedLayerId} />
              </div>

              {/* Presets Library */}
              <SceneDashboard />
            </div>
          )}

          {/* Tab: Dedicated 3D View */}
          {activeTab === 'twin' && (
            <div className="w-full h-[calc(100dvh-200px)] min-h-[380px]">
              <DigitalTwin className="h-full" />
            </div>
          )}

          {/* Tab: Layers */}
          {activeTab === 'layers' && (
            <div className="space-y-4">
              <LayerManager selectedLayerId={selectedLayerId} setSelectedLayerId={setSelectedLayerId} />
              <LayerInspector selectedLayerId={selectedLayerId} />
            </div>
          )}

          {/* Tab: Presets */}
          {activeTab === 'presets' && (
            <div>
              <SceneDashboard />
            </div>
          )}
        </div>
      </main>

      {/* Sticky Bottom Action Bar */}
      <ActionBar />
    </div>
  );
}

export default App;
