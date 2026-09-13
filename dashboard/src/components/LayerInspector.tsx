import React from 'react';
import { useSceneStore } from '../store/sceneStore';

export const LayerInspector: React.FC<{ selectedLayerId: string | null }> = ({ selectedLayerId }) => {
  const { activeLayers, updateLayer } = useSceneStore();
  
  if (!selectedLayerId) {
    return (
      <div className="bg-zinc-900 p-4 rounded-lg flex items-center justify-center text-zinc-500">
        Select a layer to inspect
      </div>
    );
  }

  const layer = activeLayers.find(l => l.id === selectedLayerId);
  if (!layer) return null;

  const handleChange = (key: string, value: any) => {
    updateLayer(selectedLayerId, { [key]: value });
  };

  const renderColorInput = (key: string, rgb: [number, number, number]) => {
    // simple hex conversion for native color picker
    const toHex = (c: number) => Math.round(c).toString(16).padStart(2, '0');
    const hex = `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;

    return (
      <div className="flex items-center space-x-2" key={key}>
        <label className="text-xs text-zinc-400 w-24">{key}</label>
        <input 
          type="color" 
          value={hex}
          onChange={(e) => {
            const h = e.target.value;
            const r = parseInt(h.substring(1, 3), 16);
            const g = parseInt(h.substring(3, 5), 16);
            const b = parseInt(h.substring(5, 7), 16);
            handleChange(key, [r, g, b]);
          }}
          className="bg-transparent border-0 cursor-pointer h-8 w-14"
        />
      </div>
    );
  };

  const renderNumberInput = (key: string, val: number, step=0.1) => (
    <div className="flex items-center space-x-2" key={key}>
      <label className="text-xs text-zinc-400 w-24">{key}</label>
      <input 
        type="number" 
        value={val}
        step={step}
        onChange={(e) => handleChange(key, parseFloat(e.target.value))}
        className="bg-zinc-800 border border-zinc-700 rounded p-1 text-sm w-24 text-white"
      />
    </div>
  );

  return (
    <div className="bg-zinc-900 p-4 rounded-lg overflow-y-auto">
      <h2 className="text-xl font-bold mb-4 uppercase">{layer.type} Settings</h2>
      
      <div className="space-y-4">
        <h3 className="text-sm border-b border-zinc-700 pb-1 text-blue-400">Base Properties</h3>
        {renderNumberInput('opacity', layer.opacity ?? 1.0)}
        {renderNumberInput('h_min', layer.h_min ?? 0.0)}
        {renderNumberInput('h_max', layer.h_max ?? 1.0)}

        <h3 className="text-sm border-b border-zinc-700 pb-1 text-blue-400 pt-4">Specific Properties</h3>
        
        {layer.type === 'solid' && (
          <>{renderColorInput('color', layer.color)}</>
        )}
        
        {layer.type === 'fire' && (
          <>
            {renderColorInput('color_start', layer.color_start)}
            {renderColorInput('color_end', layer.color_end)}
            {renderNumberInput('cooling', layer.cooling)}
            {renderNumberInput('sparking', layer.sparking)}
          </>
        )}

        {layer.type === 'clip' && (
          <div className="flex flex-col space-y-2">
            <label className="text-xs text-zinc-400">filename</label>
            <input 
              type="text" 
              value={layer.filename}
              onChange={(e) => handleChange('filename', e.target.value)}
              className="bg-zinc-800 border border-zinc-700 rounded p-1 text-sm text-white"
            />
            <div className="flex items-center space-x-2 mt-2">
              <input type="checkbox" checked={layer.transparent} onChange={(e) => handleChange('transparent', e.target.checked)} />
              <label className="text-xs text-zinc-400">transparent</label>
            </div>
          </div>
        )}
        {/* Add more inputs as needed for other types */}
      </div>
    </div>
  );
};
