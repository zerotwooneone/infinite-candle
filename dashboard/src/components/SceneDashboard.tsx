import React from 'react';
import { useSceneStore, FACTORY_PRESETS } from '../store/sceneStore';
import { LayerConfig } from '../types/schemas';

const generateId = () => Math.random().toString(36).substring(2, 9);

export const SceneDashboard: React.FC = () => {
  const { setActiveLayers, savedScenes } = useSceneStore();

  const handleLoad = (layers: any[]) => {
    // Re-generate IDs to avoid mutability issues
    const newLayers = layers.map((l: any) => ({ ...l, id: generateId() }));
    setActiveLayers(newLayers);
  };

  return (
    <div className="bg-zinc-900 p-4 rounded-lg overflow-y-auto max-h-[400px]">
      <h2 className="text-xl font-bold mb-4">Scene Presets</h2>
      
      <h3 className="text-sm text-zinc-400 mb-2 uppercase">Factory Presets</h3>
      <div className="grid grid-cols-2 gap-2 mb-6">
        {FACTORY_PRESETS.map((preset, i) => (
          <button
            key={i}
            className="bg-zinc-800 hover:bg-zinc-700 text-sm p-3 rounded text-left transition-colors"
            onClick={() => handleLoad(preset.layers)}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <h3 className="text-sm text-zinc-400 mb-2 uppercase">My Saved Scenes</h3>
      {savedScenes.length === 0 ? (
        <p className="text-xs text-zinc-600">No saved scenes yet.</p>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          {savedScenes.map((scene, i) => (
            <button
              key={i}
              className="bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-sm p-3 rounded text-left transition-colors"
              onClick={() => handleLoad(scene.layers)}
            >
              {scene.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};
