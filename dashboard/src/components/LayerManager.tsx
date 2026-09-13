import React from 'react';
import { useSceneStore } from '../store/sceneStore';
import { ChevronUp, ChevronDown, Trash2 } from 'lucide-react';
import { LayerConfig } from '../types/schemas';

export const LayerManager: React.FC<{
  selectedLayerId: string | null;
  setSelectedLayerId: (id: string | null) => void;
}> = ({ selectedLayerId, setSelectedLayerId }) => {
  const { activeLayers, moveLayer, removeLayer, addLayer } = useSceneStore();

  const handleAdd = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const type = e.target.value;
    if (!type) return;
    
    const id = Math.random().toString(36).substring(2, 9);
    let newLayer: any = { id, type, opacity: 1.0, h_min: 0.0, h_max: 1.0 };
    
    if (type === 'solid') newLayer = { ...newLayer, color: [255, 255, 255] };
    if (type === 'fire') newLayer = { ...newLayer, cooling: 0.15, sparking: 0.3, color_start: [255,0,0], color_end: [255,255,0] };
    if (type === 'clip') newLayer = { ...newLayer, filename: '', transparent: false };
    
    addLayer(newLayer as LayerConfig);
    setSelectedLayerId(id);
    e.target.value = '';
  };

  return (
    <div className="bg-zinc-900 p-4 rounded-lg flex flex-col h-full">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Active Layers</h2>
        <select 
          onChange={handleAdd} 
          className="bg-zinc-800 text-sm p-1 rounded border border-zinc-700"
          defaultValue=""
        >
          <option value="" disabled>+ Add Layer</option>
          <option value="solid">Solid</option>
          <option value="fire">Fire</option>
          <option value="lava">Lava Lamp</option>
          <option value="gol">Game of Life</option>
          <option value="stripes">Stripes</option>
          <option value="snow">Snow</option>
          <option value="clip">Clip</option>
        </select>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2">
        {/* Render bottom to top conceptually, but list top-to-bottom for UI? Let's list naturally. Index 0 is background, so index 0 at bottom. */}
        {[...activeLayers].reverse().map((layer) => {
          // reverse index mapping
          const actualIndex = activeLayers.findIndex(l => l.id === layer.id);
          
          return (
            <div 
              key={layer.id}
              className={`p-3 rounded border ${selectedLayerId === layer.id ? 'border-blue-500 bg-zinc-800' : 'border-zinc-700 bg-zinc-800/50'} flex justify-between items-center cursor-pointer`}
              onClick={() => setSelectedLayerId(layer.id)}
            >
              <div className="flex flex-col">
                <span className="font-semibold text-sm uppercase">{layer.type}</span>
                <span className="text-xs text-zinc-500">Idx: {actualIndex}</span>
              </div>
              
              <div className="flex items-center space-x-1">
                <button 
                  className="p-1 hover:bg-zinc-700 rounded text-zinc-400 hover:text-white"
                  onClick={(e) => { e.stopPropagation(); moveLayer(layer.id, 'up'); }}
                  disabled={actualIndex === activeLayers.length - 1}
                >
                  <ChevronUp size={16} />
                </button>
                <button 
                  className="p-1 hover:bg-zinc-700 rounded text-zinc-400 hover:text-white"
                  onClick={(e) => { e.stopPropagation(); moveLayer(layer.id, 'down'); }}
                  disabled={actualIndex === 0}
                >
                  <ChevronDown size={16} />
                </button>
                <button 
                  className="p-1 hover:bg-red-500/20 rounded text-zinc-400 hover:text-red-500 ml-2"
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
          <div className="text-center text-zinc-600 text-sm py-8">No active layers</div>
        )}
      </div>
    </div>
  );
};
