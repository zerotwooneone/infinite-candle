import React, { useState } from 'react';
import { useSceneStore, FACTORY_PRESETS } from '../store/sceneStore';
import { Sparkles, Bookmark, BookmarkPlus, Trash2, Check } from 'lucide-react';

const generateId = () => Math.random().toString(36).substring(2, 9);

const PRESET_ICONS: Record<string, string> = {
  'Alien': '🛸',
  'Fireworks 1': '🎆',
  'Candy Cane': '🍬',
  'Candle': '🕯️',
  'Snow': '❄️',
  'Lava Lamp': '🫧',
  'Geometric Wireframes': '📐',
  'Plasma': '⚡',
  'Chrome': '💿',
  "Rubik's Cube": '🎲',
  'Lighthouse': '🏮'
};

export const SceneDashboard: React.FC = () => {
  const { setActiveLayers, savedScenes, activeLayers, saveScene, removeSavedScene } = useSceneStore();
  const [newSceneName, setNewSceneName] = useState('');
  const [loadedPreset, setLoadedPreset] = useState<string | null>(null);

  const handleLoad = (presetName: string, layers: any[]) => {
    // Re-generate IDs to avoid mutability issues
    const newLayers = layers.map((l: any) => ({ ...l, id: generateId() }));
    setActiveLayers(newLayers);
    setLoadedPreset(presetName);
    setTimeout(() => setLoadedPreset(null), 1500);
  };

  const handleSaveScene = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newSceneName.trim();
    if (!trimmed) return;
    saveScene(trimmed);
    setNewSceneName('');
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 p-4 sm:p-5 rounded-xl shadow-sm flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-amber-400" />
          <h2 className="text-base sm:text-lg font-bold text-white">Scene Presets</h2>
        </div>
        <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 border border-zinc-700">
          {FACTORY_PRESETS.length} Factory
        </span>
      </div>
      
      {/* Scrollable Container */}
      <div className="space-y-5 max-h-[380px] sm:max-h-[460px] overflow-y-auto pr-1">
        {/* Factory Presets Grid */}
        <div>
          <h3 className="text-xs font-semibold text-zinc-400 mb-2.5 uppercase tracking-wider flex items-center gap-1.5">
            <span>✨</span> Factory Library
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-2 xl:grid-cols-3 gap-2 sm:gap-2.5">
            {FACTORY_PRESETS.map((preset, i) => {
              const icon = PRESET_ICONS[preset.name] || '💡';
              const isJustLoaded = loadedPreset === preset.name;

              return (
                <button
                  key={i}
                  className={`p-2.5 sm:p-3 rounded-xl text-left font-medium transition-all active:scale-95 flex items-center justify-between border min-h-[44px] ${
                    isJustLoaded
                      ? 'bg-blue-600/30 border-blue-500 text-white ring-1 ring-blue-500/50'
                      : 'bg-zinc-950/60 hover:bg-zinc-800/80 border-zinc-800 hover:border-zinc-700 text-zinc-300 hover:text-white'
                  }`}
                  onClick={() => handleLoad(preset.name, preset.layers)}
                >
                  <div className="flex items-center gap-2 truncate">
                    <span className="text-base shrink-0">{icon}</span>
                    <span className="text-xs sm:text-sm font-semibold truncate">{preset.name}</span>
                  </div>
                  {isJustLoaded && <Check size={14} className="text-blue-400 shrink-0 ml-1" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Custom Saved Scenes */}
        <div className="pt-2 border-t border-zinc-800/80">
          <div className="flex items-center justify-between mb-2.5">
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <Bookmark className="w-3.5 h-3.5 text-blue-400" /> My Saved Scenes
            </h3>
            {savedScenes.length > 0 && (
              <span className="text-[11px] text-zinc-500 font-mono">
                {savedScenes.length} saved
              </span>
            )}
          </div>

          {/* Save Current Scene Form */}
          <form onSubmit={handleSaveScene} className="flex gap-2 mb-3">
            <input
              type="text"
              value={newSceneName}
              onChange={(e) => setNewSceneName(e.target.value)}
              placeholder="Name current scene..."
              className="flex-1 bg-zinc-950/70 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs sm:text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              disabled={activeLayers.length === 0}
            />
            <button
              type="submit"
              disabled={!newSceneName.trim() || activeLayers.length === 0}
              className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all active:scale-95 shrink-0"
            >
              <BookmarkPlus size={14} />
              <span className="hidden sm:inline">Save</span>
            </button>
          </form>

          {/* Saved Scenes List */}
          {savedScenes.length === 0 ? (
            <div className="p-3 rounded-lg bg-zinc-950/40 border border-zinc-800/60 text-center text-xs text-zinc-500">
              No custom scenes saved yet. Compose layers and click Save to store them locally.
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {savedScenes.map((scene, i) => (
                <div
                  key={i}
                  className="bg-zinc-950/60 border border-zinc-800 hover:border-zinc-700 rounded-xl p-2.5 flex items-center justify-between gap-2 transition-colors"
                >
                  <button
                    className="flex-1 text-left text-xs sm:text-sm text-zinc-300 hover:text-white font-medium truncate"
                    onClick={() => handleLoad(scene.name, scene.layers)}
                  >
                    🎨 {scene.name}
                  </button>
                  <button
                    aria-label={`Delete saved scene ${scene.name}`}
                    className="p-1.5 rounded-md text-zinc-500 hover:text-red-400 hover:bg-red-500/10 active:scale-95 transition-all shrink-0"
                    onClick={() => removeSavedScene(scene.name)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
