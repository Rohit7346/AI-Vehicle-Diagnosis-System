import React from 'react';
import { Activity, ShieldCheck, Cpu } from 'lucide-react';

export default function Navbar({ systemHealth }) {
  const isHealthy = systemHealth?.status === 'healthy';
  const pipelineReady = systemHealth?.models?.pipeline_ready;

  return (
    <header className="h-16 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-6 sticky top-0 z-50">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-blue-600/20 text-blue-400 rounded-lg border border-blue-500/30">
          <Activity className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-base font-bold tracking-wide text-white">
            AI Multimodal Vehicle Diagnosis System
          </h1>
          <p className="text-xs text-slate-400">Final-Year Research Engineering Platform</p>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* API Status Badge */}
        <div className="flex items-center space-x-2 px-3 py-1 bg-slate-800/80 rounded-full border border-slate-700 text-xs">
          <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
          <span className="text-slate-300">
            Backend: {isHealthy ? 'Online' : 'Offline'}
          </span>
        </div>

        {/* Model Pipeline Readiness */}
        <div className="flex items-center space-x-2 px-3 py-1 bg-slate-800/80 rounded-full border border-slate-700 text-xs">
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
          <span className="text-slate-300">
            AI Pipeline: {pipelineReady ? (
              <span className="text-emerald-400 font-medium">Ready</span>
            ) : (
              <span className="text-amber-400 font-medium">Weights Missing</span>
            )}
          </span>
        </div>

        <div className="flex items-center space-x-1.5 px-2.5 py-1 bg-blue-950/60 border border-blue-800/50 rounded text-xs text-blue-300">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>v1.0.0</span>
        </div>
      </div>
    </header>
  );
}
