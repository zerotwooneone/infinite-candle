import React, { useState } from 'react';
import { useSceneStore } from '../store/sceneStore';
import { API_BASE } from '../config';
import { Play, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

export const ActionBar: React.FC = () => {
  const { activeLayers } = useSceneStore();
  const [status, setStatus] = useState<'idle' | 'pushing' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');

  const handlePush = async () => {
    setStatus('pushing');
    setErrorMessage('');
    try {
      // Keep layer `id` so the backend Engine can match and preserve
      // running plugin instances across scene pushes.
      const payload = activeLayers;
      
      const res = await fetch(`${API_BASE}/scene`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layers: payload })
      });
      
      if (res.ok) {
        setStatus('success');
        setTimeout(() => setStatus('idle'), 2500);
      } else {
        setStatus('error');
        setErrorMessage(`Server error (${res.status})`);
        setTimeout(() => setStatus('idle'), 3500);
      }
    } catch (e: any) {
      console.error(e);
      setStatus('error');
      setErrorMessage('Network error');
      setTimeout(() => setStatus('idle'), 3500);
    }
  };

  return (
    <footer className="bg-zinc-900/95 backdrop-blur-md border-t border-zinc-800/90 px-4 sm:px-6 py-3 fixed bottom-0 left-0 right-0 z-40 pb-[calc(0.75rem+env(safe-area-inset-bottom,0px))] shadow-2xl">
      <div className="max-w-[1920px] mx-auto flex items-center justify-between gap-3">
        {/* Left: Layer summary & status feedback */}
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <div className="flex items-center gap-1.5 text-xs sm:text-sm text-zinc-300 font-medium shrink-0">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            <span>
              {activeLayers.length} {activeLayers.length === 1 ? 'Layer' : 'Layers'} Active
            </span>
          </div>

          {/* Status Message */}
          {status === 'success' && (
            <span className="flex items-center gap-1 text-xs text-emerald-400 font-medium animate-fadeIn">
              <CheckCircle2 size={14} className="shrink-0" />
              <span className="hidden sm:inline">Pushed to Candle!</span>
            </span>
          )}
          {status === 'error' && (
            <span className="flex items-center gap-1 text-xs text-red-400 font-medium animate-fadeIn truncate">
              <AlertCircle size={14} className="shrink-0" />
              <span>{errorMessage || 'Failed to push'}</span>
            </span>
          )}
        </div>
        
        {/* Right: Push to Candle Action */}
        <div className="flex items-center gap-3 shrink-0">
          <button 
            onClick={handlePush}
            disabled={status === 'pushing'}
            className="bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:opacity-50 text-white px-5 sm:px-7 py-2.5 sm:py-3 rounded-full font-bold text-xs sm:text-sm flex items-center gap-2 shadow-lg shadow-blue-900/30 transition-all active:scale-95 cursor-pointer touch-manipulation"
          >
            {status === 'pushing' ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                <span>Pushing...</span>
              </>
            ) : status === 'success' ? (
              <>
                <CheckCircle2 size={16} className="text-emerald-300" />
                <span>Pushed!</span>
              </>
            ) : (
              <>
                <Play size={16} fill="currentColor" />
                <span>Push to Candle</span>
              </>
            )}
          </button>
        </div>
      </div>
    </footer>
  );
};
