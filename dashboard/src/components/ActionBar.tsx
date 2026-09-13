import React, { useState } from 'react';
import { useSceneStore } from '../store/sceneStore';
import { Play } from 'lucide-react';

export const ActionBar: React.FC = () => {
  const { activeLayers } = useSceneStore();
  const [status, setStatus] = useState<string>('');

  const handlePush = async () => {
    setStatus('Pushing...');
    try {
      // Clean up internal IDs before sending to backend
      const payload = activeLayers.map(({ id, ...rest }) => rest);
      
      const res = await fetch('http://10.0.217.7:8000/scene', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layers: payload })
      });
      
      if (res.ok) {
        setStatus('Success!');
        setTimeout(() => setStatus(''), 2000);
      } else {
        setStatus('Error!');
      }
    } catch (e) {
      console.error(e);
      setStatus('Network Error!');
    }
  };

  return (
    <div className="bg-zinc-900 border-t border-zinc-800 p-4 flex justify-between items-center fixed bottom-0 left-0 right-0 z-50">
      <div className="text-zinc-400 text-sm">
        {activeLayers.length} Layers Active
      </div>
      
      <div className="flex items-center space-x-4">
        {status && <span className="text-sm text-green-400">{status}</span>}
        <button 
          onClick={handlePush}
          className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-full font-bold flex items-center shadow-lg transition-transform hover:scale-105"
        >
          <Play size={18} className="mr-2" fill="currentColor" />
          Push to Candle
        </button>
      </div>
    </div>
  );
};
