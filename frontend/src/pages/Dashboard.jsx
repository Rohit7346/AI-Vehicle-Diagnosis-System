import React from 'react';
import { Activity, Cpu, Stethoscope, BookOpen, AlertTriangle, ArrowRight, ShieldCheck, CheckCircle2 } from 'lucide-react';
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
          Final-year engineering workstation for multimodal vehicle subassembly inspection and defect classification.
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Subassembly Classes</span>
            <Cpu className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold text-white">10 Subsystems</div>
          <p className="text-[11px] text-slate-400 mt-1">Engine bay, brake rotors, calipers, tires, battery, etc.</p>
        </div>

        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Defect Classes</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-white">12 Fault Types</div>
          <p className="text-[11px] text-slate-400 mt-1">Leaks (oil/coolant/brake), cracks, rust, wear, grooving, bulges</p>
        </div>

        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Architecture State</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-white">Modular Monorepo</div>
          <p className="text-[11px] text-slate-400 mt-1">Strict decoupled layers: ML Core &bull; FastAPI &bull; React</p>
        </div>
      </div>

      {/* Live System & Model Health */}
      <SystemStatus healthData={healthData} onRefresh={onRefresh} />

      {/* Model Readiness Guidance Alert */}
      {(!compLoaded || !faultLoaded) && (
        <div className="p-4 bg-amber-950/30 border border-amber-800/60 rounded-xl flex items-start space-x-3 text-xs text-amber-200">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-amber-100 text-sm">Model Weights Pending Training</h4>
            <p className="text-amber-300">
              In accordance with academic integrity guidelines, mock predictions are strictly disabled. To enable live inference:
            </p>
            <ol className="list-decimal list-inside space-y-0.5 text-amber-300/90 pt-1 font-mono text-[11px]">
              <li>Collect high-res inspection images in <code className="bg-amber-900/40 px-1 py-0.5 rounded">data/raw/images/</code></li>
              <li>Preprocess & annotate samples in <code className="bg-amber-900/40 px-1 py-0.5 rounded">data/processed/</code></li>
              <li>Execute <code className="bg-amber-900/40 px-1 py-0.5 rounded">python -m ml.training.train_components</code></li>
              <li>Execute <code className="bg-amber-900/40 px-1 py-0.5 rounded">python -m ml.training.train_faults</code></li>
            </ol>
          </div>
        </div>
      )}

      {/* Quick Access Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button
          onClick={() => setActiveTab('diagnostics')}
          className="p-5 bg-gradient-to-br from-slate-800/60 to-slate-900/60 border border-slate-700/80 hover:border-blue-500 rounded-xl text-left transition group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="p-2.5 bg-blue-600/20 text-blue-400 rounded-lg">
              <Stethoscope className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-blue-400 group-hover:translate-x-1 transition" />
          </div>
          <h3 className="text-sm font-bold text-white mb-1">Launch Diagnostic Ingestion</h3>
          <p className="text-xs text-slate-400">
            Upload an inspection photo to execute dual-stage subassembly localization and visible fault diagnosis.
          </p>
        </button>

        <button
          onClick={() => setActiveTab('models')}
          className="p-5 bg-gradient-to-br from-slate-800/60 to-slate-900/60 border border-slate-700/80 hover:border-blue-500 rounded-xl text-left transition group"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="p-2.5 bg-purple-600/20 text-purple-400 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-purple-400 group-hover:translate-x-1 transition" />
          </div>
          <h3 className="text-sm font-bold text-white mb-1">Inspect Model Weights & Architecture</h3>
          <p className="text-xs text-slate-400">
            Review checkpoint file locations, YOLO configuration definitions, and dataset annotation compliance.
          </p>
        </button>
      </div>
    </div>
  );
}
