import React from 'react';
import { useSceneStore } from '../store/sceneStore';
import { ChevronUp, ChevronDown, Trash2, Plus, Layers, Flame, Palette, Sparkles } from 'lucide-react';
import { LayerConfig } from '../types/schemas';

export const LayerManager: React.FC<{
  selectedLayerId: string | null;
  setSelectedLayerId: (id: string | null) => void;
}> = ({ selectedLayerId, setSelectedLayerId }) => {
  const { activeLayers, moveLayer, removeLayer, addLayer, plugins } = useSceneStore();

  const handleAdd = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const type = e.target.value;
    if (!type) return;
    
    const id = Math.random().toString(36).substring(2, 9);

    // Plugin layers own their params entirely — only their declared defaults.
    const pluginDef = plugins[type];
    if (pluginDef) {
      addLayer({ id, type, ...pluginDef.defaults } as LayerConfig);
      setSelectedLayerId(id);
      e.target.value = '';
      return;
    }

    let newLayer: any = { id, type, opacity: 1.0, h_min: 0.0, h_max: 1.0 };
    
    if (type === 'solid') newLayer = { ...newLayer, color: [255, 255, 255] };
    if (type === 'fire') newLayer = { ...newLayer, cooling: 0.15, sparking: 0.3, color_start: [255, 180, 40], color_end: [255, 0, 0] };
    if (type === 'stripes') newLayer = { ...newLayer, color_a: [255, 0, 0], color_b: [255, 255, 255], angle: 45.0, width: 0.5, speed: 1.0 };
    if (type === 'lava') newLayer = { ...newLayer, color: [255, 0, 0], bg_color: [0, 200, 0], blob_count: 6, speed: 0.8 };
    if (type === 'clip') newLayer = { ...newLayer, filename: '', transparent: false };
    if (type === 'alien') newLayer = { ...newLayer, speed: 1.0, transparent: true, ship_color_1: [50, 150, 255], ship_color_2: [0, 255, 50], beam_color: [100, 255, 255] };
    if (type === 'fireworks') newLayer = { ...newLayer, launch_rate: 1.2, burst_height: 0.78, explosion_size: 0.02, max_rockets: 2, brightness: 1.0 };
    
    addLayer(newLayer as LayerConfig);
    setSelectedLayerId(id);
    e.target.value = '';
  };

  const getLayerIcon = (type: string) => {
    switch (type) {
      case 'fire':
        return <Flame className="w-4 h-4 text-orange-400" />;
      case 'solid':
        return <Palette className="w-4 h-4 text-emerald-400" />;
      default:
        return <Sparkles className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 p-4 sm:p-5 rounded-xl flex flex-col h-full shadow-sm">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-2 mb-4 pb-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-blue-400" />
          <h2 className="text-base sm:text-lg font-bold text-white">Active Layers</h2>
          <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 border border-zinc-700 font-mono">
            {activeLayers.length}
          </span>
        </div>

        {/* Add Layer selector with touch friendly sizing */}
        <div className="relative">
          <select 
            onChange={handleAdd} 
            className="bg-zinc-800 hover:bg-zinc-750 text-white text-xs sm:text-sm py-1.5 px-3 pr-7 rounded-lg border border-zinc-700 focus:outline-none focus:ring-2 focus:ring-blue-500 font-medium cursor-pointer transition-colors appearance-none"
            defaultValue=""
          >
            <option value="" disabled>+ Add Layer</option>
            <option value="fire">🔥 Fire</option>
            <option value="solid">🎨 Solid Color</option>
            <option value="stripes">💈 Stripes</option>
            <option value="lava">🫧 Lava Lamp</option>
            <option value="alien">🛸 Alien</option>
            <option value="fireworks">🎆 Fireworks</option>
            <option value="clip">🎬 Clip (.npy)</option>
            <option value="gol">👾 Game of Life</option>
            <option value="snow">❄️ Snow</option>
            {Object.keys(plugins).length > 0 && (
              <optgroup label="Plugins">
                {Object.values(plugins).map((p) => (
                  <option key={p.id} value={p.id}>🧩 {p.name}</option>
                ))}
              </optgroup>
            )}
          </select>
          <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-zinc-400">
            <Plus size={14} />
          </div>
        </div>
      </div>

      {/* Layer List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-[180px] max-h-[380px] sm:max-h-[500px]">
        {[...activeLayers].reverse().map((layer) => {
          const actualIndex = activeLayers.findIndex(l => l.id === layer.id);
          const isSelected = selectedLayerId === layer.id;
          
          return (
            <div 
              key={layer.id}
              className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-2 ${
                isSelected 
                  ? 'border-blue-500 bg-blue-950/20 shadow-md ring-1 ring-blue-500/30' 
                  : 'border-zinc-800 bg-zinc-950/40 hover:bg-zinc-800/60 hover:border-zinc-700'
              }`}
              onClick={() => setSelectedLayerId(layer.id)}
            >
              {/* Layer Info */}
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="p-1.5 rounded-lg bg-zinc-800/80 border border-zinc-700/60 shrink-0">
                  {getLayerIcon(layer.type)}
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="font-semibold text-xs sm:text-sm text-white uppercase tracking-wider truncate">
                    {plugins[layer.type]?.name || layer.type}
                  </span>
                  <span className="text-[11px] text-zinc-400 font-mono">
                    Layer #{actualIndex + 1}
                    {layer.opacity !== undefined && ` • ${Math.round(layer.opacity * 100)}% opac`}
                  </span>
                </div>
              </div>
              
              {/* Layer Actions with large accessible touch targets */}
              <div className="flex items-center space-x-1 shrink-0">
                <button 
                  aria-label="Move layer up"
                  className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-700 active:bg-zinc-600 active:scale-95 transition-all disabled:opacity-20 disabled:pointer-events-none"
                  onClick={(e) => { e.stopPropagation(); moveLayer(layer.id, 'up'); }}
                  disabled={actualIndex === activeLayers.length - 1}
                >
                  <ChevronUp size={18} />
                </button>
                <button 
                  aria-label="Move layer down"
                  className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-700 active:bg-zinc-600 active:scale-95 transition-all disabled:opacity-20 disabled:pointer-events-none"
                  onClick={(e) => { e.stopPropagation(); moveLayer(layer.id, 'down'); }}
                  disabled={actualIndex === 0}
                >
                  <ChevronDown size={18} />
                </button>
                <button 
                  aria-label="Delete layer"
                  className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center rounded-lg text-zinc-400 hover:text-red-400 hover:bg-red-500/10 active:bg-red-500/20 active:scale-95 transition-all ml-1"
                  onClick={(e) => { 
                    e.stopPropagation(); 
                    removeLayer(layer.id);
                    if (selectedLayerId === layer.id) setSelectedLayerId(null);
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>
          );
        })}

        {activeLayers.length === 0 && (
          <div className="text-center py-10 px-4 border border-dashed border-zinc-800 rounded-xl">
            <Layers className="w-8 h-8 mx-auto mb-2 text-zinc-600" />
            <p className="text-sm font-medium text-zinc-400">No active layers</p>
            <p className="text-xs text-zinc-500 mt-1 max-w-xs mx-auto">
              Add a layer above or choose a preset from the Presets panel to compose light patterns.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
