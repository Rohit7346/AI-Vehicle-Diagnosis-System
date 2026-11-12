import React from 'react';
import { Cpu, Stethoscope, Layers, AlertTriangle, ArrowRight, ShieldCheck, CheckCircle2 } from 'lucide-react';
import SystemStatus from '../components/SystemStatus';

export default function Dashboard({ healthData, onRefresh, setActiveTab }) {
  const compLoaded = healthData?.models?.component_detector?.loaded;
  const faultLoaded = healthData?.models?.fault_detector?.loaded;

  return (
    <div className="space-y-6">
      {/* Title & Introduction */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">System Operations & Research Overview</h2>
        <p className="text-xs text-slate-400 mt-1">
          Multimodal vehicle fault diagnosis workstation: YOLOv8 component perception and defect classification.
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Trained Component Classes</span>
            <Layers className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-white">3 Subsystems</div>
          <p className="text-[11px] text-slate-400 mt-1">Engine Bay, 12V Battery, Radiator (YOLOv8 Nano)</p>
        </div>

        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Perception Status</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">Component Ready</div>
          <p className="text-[11px] text-slate-400 mt-1">YOLOv8n-VehicleComponents-3Class deployed</p>
        </div>

        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Fault Pipeline</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-400">Pending Stage 5/6</div>
          <p className="text-[11px] text-slate-400 mt-1">Defect detector weights in scheduled development</p>
        </div>
      </div>

      {/* Live System & Model Health */}
      <SystemStatus healthData={healthData} onRefresh={onRefresh} />

      {/* Quick Access Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={() => setActiveTab('components')}
          className="p-5 bg-gradient-to-br from-blue-950/40 to-slate-900/60 border border-blue-800/60 hover:border-blue-500 rounded-xl text-left transition group shadow-sm hover:shadow-blue-500/10"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="p-2.5 bg-blue-600/20 text-blue-400 rounded-lg">
              <Layers className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-blue-400 group-hover:translate-x-1 transition" />
          </div>
          <h3 className="text-sm font-bold text-white mb-1">Launch Component Detection</h3>
          <p className="text-xs text-slate-400">
            Real-time localization of engine bay, battery, and radiator with interactive bounding box overlays.
          </p>
        </button>

        <button
          onClick={() => setActiveTab('diagnostics')}
          className="p-5 bg-gradient-to-br from-slate-800/60 to-slate-900/60 border border-slate-700/80 hover:border-slate-500 rounded-xl text-left transition group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="p-2.5 bg-slate-800 text-slate-400 rounded-lg">
              <Stethoscope className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-slate-300 group-hover:translate-x-1 transition" />
          </div>
          <h3 className="text-sm font-bold text-white mb-1">Subsystem Diagnostic Console</h3>
          <p className="text-xs text-slate-400">
            End-to-end multimodal defect classification pipeline (currently returning 503 until Stage 5/6).
          </p>
        </button>

        <button
          onClick={() => setActiveTab('models')}
          className="p-5 bg-gradient-to-br from-slate-800/60 to-slate-900/60 border border-slate-700/80 hover:border-purple-500 rounded-xl text-left transition group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="p-2.5 bg-purple-600/20 text-purple-400 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-purple-400 group-hover:translate-x-1 transition" />
          </div>
          <h3 className="text-sm font-bold text-white mb-1">Inspect Model Weights</h3>
          <p className="text-xs text-slate-400">
            Review checkpoint file locations, YOLOv8n parameters, and validation metrics.
          </p>
        </button>
      </div>
    </div>
  );
}
