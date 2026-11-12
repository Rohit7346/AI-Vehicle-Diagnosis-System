import React from 'react';
import { CheckCircle2, AlertTriangle, Cpu, Terminal } from 'lucide-react';

export default function SystemStatus({ healthData, onRefresh }) {
  const compModel = healthData?.models?.component_detector;
  const faultModel = healthData?.models?.fault_detector;
  const sys = healthData?.system;

  return (
    <div className="bg-slate-800/40 border border-slate-800 rounded-xl p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-blue-400" />
          <span>System Readiness & Hardware Acceleration</span>
        </h3>
        <button
          onClick={onRefresh}
          className="text-xs text-blue-400 hover:text-blue-300 font-medium transition"
        >
          Refresh Status
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Python Environment */}
        <div className="p-3.5 bg-slate-900/60 rounded-lg border border-slate-800/80">
          <span className="text-xs text-slate-400 block mb-1">Python Runtime</span>
          <span className="text-sm font-mono font-medium text-slate-200">
            v{sys?.python_version || 'Unknown'}
          </span>
        </div>

        {/* Compute Device */}
        <div className="p-3.5 bg-slate-900/60 rounded-lg border border-slate-800/80">
          <span className="text-xs text-slate-400 block mb-1">Compute Target</span>
          <div className="flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-slate-400" />
            <span className="text-sm font-mono text-slate-200 uppercase">
              {sys?.device || 'cpu'} {sys?.cuda_available ? '(CUDA)' : '(CPU Only)'}
            </span>
          </div>
        </div>

        {/* Component Detector Weights */}
        <div className="p-3.5 bg-slate-900/60 rounded-lg border border-slate-800/80">
          <span className="text-xs text-slate-400 block mb-1">Component Weights</span>
          <div className="flex items-center space-x-1.5">
            {compModel?.loaded ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-emerald-400 font-medium">Loaded</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="text-xs text-amber-400 font-medium">Pending Training</span>
              </>
            )}
          </div>
        </div>

        {/* Visible Fault Detector Weights */}
        <div className="p-3.5 bg-slate-900/60 rounded-lg border border-slate-800/80">
          <span className="text-xs text-slate-400 block mb-1">Fault Weights</span>
          <div className="flex items-center space-x-1.5">
            {faultModel?.loaded ? (
              <>
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-emerald-400 font-medium">Loaded</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span className="text-xs text-amber-400 font-medium">Pending Training</span>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
