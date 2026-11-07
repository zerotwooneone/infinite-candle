import React from 'react';
import { useSceneStore } from '../store/sceneStore';
import { Sliders, Sparkles, Layers } from 'lucide-react';

export const LayerInspector: React.FC<{ selectedLayerId: string | null }> = ({ selectedLayerId }) => {
  const { activeLayers, updateLayer } = useSceneStore();
  
  if (!selectedLayerId) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 p-6 rounded-xl flex flex-col items-center justify-center text-zinc-500 text-center min-h-[160px] shadow-sm">
        <Layers className="w-8 h-8 mb-2 opacity-40 text-blue-400" />
        <p className="text-sm font-medium text-zinc-400">No Layer Selected</p>
        <p className="text-xs text-zinc-600 mt-1">Tap a layer in the Active Layers list to inspect and tune its properties.</p>
      </div>
    );
  }

  const layer = activeLayers.find(l => l.id === selectedLayerId);
  if (!layer) return null;

  const handleChange = (key: string, value: any) => {
    updateLayer(selectedLayerId, { [key]: value });
  };

  const renderColorInput = (key: string, rgb?: [number, number, number], label?: string) => {
    const validRgb = Array.isArray(rgb) && rgb.length === 3 ? rgb : [255, 255, 255];
    const toHex = (c: number) => Math.min(255, Math.max(0, Math.round(c))).toString(16).padStart(2, '0');
    const hex = `#${toHex(validRgb[0])}${toHex(validRgb[1])}${toHex(validRgb[2])}`;

    return (
      <div className="flex items-center justify-between py-2 border-b border-zinc-800/80 gap-3" key={key}>
        <label className="text-xs font-medium text-zinc-300 capitalize">{label || key.replace('_', ' ')}</label>
        <div className="flex items-center gap-2.5">
          <label className="relative cursor-pointer flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700/80 px-2.5 py-1 rounded-lg border border-zinc-700 transition-colors">
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
              className="opacity-0 absolute inset-0 w-full h-full cursor-pointer"
            />
            <span
              className="w-4 h-4 rounded-full border border-white/20 shadow-sm shrink-0"
              style={{ backgroundColor: hex }}
            />
            <span className="text-xs text-zinc-300 font-mono tracking-wider">{hex.toUpperCase()}</span>
          </label>
        </div>
      </div>
    );
  };

  const renderNumberInput = (
    key: string,
    val: number | undefined,
    min = 0.0,
    max = 1.0,
    step = 0.05,
    label?: string
  ) => {
    const currentVal = typeof val === 'number' && !isNaN(val) ? val : min;
    const hasSlider = typeof min === 'number' && typeof max === 'number';

    return (
      <div className="flex flex-col sm:flex-row sm:items-center justify-between py-2 border-b border-zinc-800/80 gap-2" key={key}>
        <div className="flex items-center justify-between sm:w-28 shrink-0">
          <label className="text-xs font-medium text-zinc-300 capitalize">{label || key.replace('_', ' ')}</label>
          <span className="sm:hidden text-xs text-zinc-400 font-mono">
            {step < 0.1 ? currentVal.toFixed(2) : currentVal}
          </span>
        </div>
        
        <div className="flex items-center gap-3 flex-1">
          {hasSlider && (
            <input 
              type="range"
              min={min}
              max={max}
              step={step}
              value={currentVal}
              onChange={(e) => handleChange(key, parseFloat(e.target.value))}
              className="w-full flex-1"
            />
          )}
          <input 
            type="number" 
            value={currentVal}
            min={min}
            max={max}
            step={step}
            onChange={(e) => {
              const num = parseFloat(e.target.value);
              if (!isNaN(num)) handleChange(key, num);
            }}
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-2 py-1 text-xs sm:text-sm w-20 text-white font-mono text-right focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>
    );
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 p-4 sm:p-5 rounded-xl flex flex-col shadow-sm">
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-blue-400" />
          <h2 className="text-base sm:text-lg font-bold text-white uppercase tracking-wide">
            {layer.type} Settings
          </h2>
        </div>
        <span className="text-xs px-2 py-0.5 rounded-full bg-blue-950/60 text-blue-400 border border-blue-800/60 font-mono">
          ID: {layer.id.substring(0, 5)}
        </span>
      </div>
      
      <div className="space-y-4 max-h-[380px] sm:max-h-[460px] overflow-y-auto pr-1">
        {/* Base Properties */}
        <div>
          <h3 className="text-xs uppercase font-semibold text-blue-400 tracking-wider mb-2 flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5" /> Base Properties
          </h3>
          <div className="bg-zinc-950/50 p-2.5 rounded-lg border border-zinc-800/80">
            {renderNumberInput('opacity', layer.opacity ?? 1.0, 0, 1, 0.05, 'Opacity')}
            {renderNumberInput('h_min', layer.h_min ?? 0.0, 0, 1, 0.02, 'Min Height')}
            {renderNumberInput('h_max', layer.h_max ?? 1.0, 0, 1, 0.02, 'Max Height')}
          </div>
        </div>

        {/* Specific Properties */}
        <div>
          <h3 className="text-xs uppercase font-semibold text-blue-400 tracking-wider mb-2">
            Layer Parameters
          </h3>
          <div className="bg-zinc-950/50 p-2.5 rounded-lg border border-zinc-800/80">
            {layer.type === 'solid' && (
              <>{renderColorInput('color', layer.color, 'Solid Color')}</>
            )}
            
            {layer.type === 'fire' && (
              <>
                {renderColorInput('color_start', layer.color_start, 'Base Flame')}
                {renderColorInput('color_end', layer.color_end, 'Flame Tip')}
                {renderNumberInput('cooling', layer.cooling ?? 0.15, 0.01, 5.0, 0.05, 'Cooling')}
                {renderNumberInput('sparking', layer.sparking ?? 0.3, 0.01, 1.0, 0.05, 'Sparking')}
              </>
            )}

            {layer.type === 'stripes' && (
              <>
                {renderColorInput('color_a', (layer as any).color_a, 'Color A')}
                {renderColorInput('color_b', (layer as any).color_b, 'Color B')}
                {renderNumberInput('angle', (layer as any).angle ?? 45, -90, 90, 1, 'Angle (°)')}
                {renderNumberInput('width', (layer as any).width ?? 0.5, 0.05, 2.0, 0.05, 'Stripe Width')}
                {renderNumberInput('speed', (layer as any).speed ?? 1.0, -5.0, 5.0, 0.1, 'Speed')}
              </>
            )}

            {layer.type === 'lava' && (
              <>
                {renderColorInput('color', (layer as any).color, 'Blob Color')}
                {renderColorInput('bg_color', (layer as any).bg_color, 'Background')}
                {renderNumberInput('blob_count', (layer as any).blob_count ?? 6, 1, 16, 1, 'Blobs')}
                {renderNumberInput('speed', (layer as any).speed ?? 0.8, 0.1, 5.0, 0.1, 'Speed')}
              </>
            )}

            {layer.type === 'alien' && (
              <>
                {renderColorInput('ship_color_1', (layer as any).ship_color_1, 'Ship Color 1')}
                {renderColorInput('ship_color_2', (layer as any).ship_color_2, 'Ship Color 2')}
                {renderColorInput('beam_color', (layer as any).beam_color, 'Beam Color')}
                {renderNumberInput('speed', (layer as any).speed ?? 1.0, 0.1, 5.0, 0.1, 'Speed')}
              </>
            )}

            {layer.type === 'fireworks' && (
              <>
                {renderNumberInput('launch_rate', (layer as any).launch_rate ?? 1.2, 0.1, 5.0, 0.1, 'Launch Rate')}
                {renderNumberInput('burst_height', (layer as any).burst_height ?? 0.78, 0.1, 1.0, 0.02, 'Burst Height')}
                {renderNumberInput('explosion_size', (layer as any).explosion_size ?? 0.02, 0.01, 0.1, 0.005, 'Explosion')}
                {renderNumberInput('brightness', (layer as any).brightness ?? 1.0, 0.1, 2.0, 0.1, 'Brightness')}
              </>
            )}

            {layer.type === 'clip' && (
              <div className="flex flex-col space-y-3 py-1">
                <div>
                  <label className="text-xs font-medium text-zinc-300 block mb-1">File Name</label>
                  <input 
                    type="text" 
                    value={layer.filename || ''} 
                    placeholder="e.g. snow_heavy.npy"
                    onChange={(e) => handleChange('filename', e.target.value)}
                    className="w-full bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-xs sm:text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
                <label className="flex items-center space-x-2.5 cursor-pointer py-1">
                  <input 
                    type="checkbox" 
                    checked={layer.transparent || false} 
                    onChange={(e) => handleChange('transparent', e.target.checked)} 
                    className="w-4 h-4 rounded bg-zinc-800 border-zinc-700 text-blue-600 focus:ring-0 focus:ring-offset-0 cursor-pointer"
                  />
                  <span className="text-xs font-medium text-zinc-300">Transparent Background</span>
                </label>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
