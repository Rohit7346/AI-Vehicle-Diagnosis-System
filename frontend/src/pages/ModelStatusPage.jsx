import React from 'react';
import { Cpu, CheckCircle2, AlertTriangle, Terminal } from 'lucide-react';

export default function ModelStatusPage({ healthData }) {
  const compModel = healthData?.models?.component_detector;
  const faultModel = healthData?.models?.fault_detector;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Model Architecture & Checkpoints</h2>
        <p className="text-xs text-slate-400 mt-1">
          Detailed status of Ultralytics YOLO vision models, training scripts, and checkpoint availability.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Component Detector Card */}
        <div className="p-5 bg-slate-800/40 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center space-x-2.5">
              <Cpu className="w-5 h-5 text-blue-400" />
              <div>
                <h3 className="text-sm font-bold text-slate-200">Vehicle Component Detector</h3>
                <span className="text-[11px] text-slate-400">Stage 1 Localization Backbone</span>
              </div>
            </div>
            {compModel?.loaded ? (
              <span className="flex items-center space-x-1 px-2.5 py-1 bg-emerald-950/60 border border-emerald-800 rounded text-xs text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Loaded</span>
              </span>
            ) : (
              <span className="flex items-center space-x-1 px-2.5 py-1 bg-amber-950/60 border border-amber-800 rounded text-xs text-amber-400">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Missing Weights</span>
              </span>
            )}
          </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Deployed Checkpoint:</span>
                <span className="font-mono text-slate-200">{compModel?.weights_path || 'ml/weights/component_detector_best.pt'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">File Exists on Disk:</span>
                <span className="font-mono text-emerald-400 font-semibold">{compModel?.weights_exist_on_disk ? 'Yes (Verified)' : 'No'}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Active Classes (3):</span>
                <span className="text-slate-200">0: engine_bay, 1: battery, 2: radiator</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Validation Performance:</span>
                <span className="font-mono text-blue-300">mAP@0.5: 0.673 | mAP50-95: 0.411</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Architecture:</span>
                <span className="text-slate-200">Ultralytics YOLOv8 Nano (3.01M params)</span>
              </div>
            </div>

            <div className="p-3 bg-slate-900/80 rounded-lg border border-slate-800">
              <span className="text-[11px] text-slate-400 font-semibold block mb-1">Inference API Endpoint:</span>
              <code className="text-xs font-mono text-blue-300">
                POST /api/v1/detect/components
              </code>
            </div>
        </div>

        {/* Fault Detector Card */}
        <div className="p-5 bg-slate-800/40 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center space-x-2.5">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              <div>
                <h3 className="text-sm font-bold text-slate-200">Visible Fault Detector</h3>
                <span className="text-[11px] text-slate-400">Stage 2 Defect Classification</span>
              </div>
            </div>
            {faultModel?.loaded ? (
              <span className="flex items-center space-x-1 px-2.5 py-1 bg-emerald-950/60 border border-emerald-800 rounded text-xs text-emerald-400">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Loaded</span>
              </span>
            ) : (
              <span className="flex items-center space-x-1 px-2.5 py-1 bg-amber-950/60 border border-amber-800 rounded text-xs text-amber-400">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Missing Weights</span>
              </span>
            )}
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Expected Checkpoint:</span>
              <span className="font-mono text-slate-200">{faultModel?.weights_path || 'ml/weights/fault_detector_best.pt'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">File Exists on Disk:</span>
              <span className="font-mono text-slate-200">{faultModel?.weights_exist_on_disk ? 'Yes' : 'No'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Target Classes:</span>
              <span className="text-slate-200">12 Physical Defect Categories</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-400">Architecture:</span>
              <span className="text-slate-200">Ultralytics YOLOv8 Medium (YOLOv8m)</span>
            </div>
          </div>

          <div className="p-3 bg-slate-900/80 rounded-lg border border-slate-800">
            <span className="text-[11px] text-slate-400 font-semibold block mb-1">Training Command:</span>
            <code className="text-xs font-mono text-amber-300">
              python -m ml.training.train_faults --epochs 120 --batch-size 16
            </code>
          </div>
        </div>
      </div>

      {/* Dataset Verification Protocol Box */}
      <div className="p-5 bg-slate-800/40 border border-slate-800 rounded-xl space-y-3">
        <h3 className="text-sm font-bold text-slate-200 flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-blue-400" />
          <span>Dataset Pre-Flight Audit Tool</span>
        </h3>
        <p className="text-xs text-slate-400">
          Before launching training, run the built-in validator to confirm image dimensions, annotation formatting, and class index boundaries:
        </p>
        <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
          <code className="text-xs font-mono text-emerald-400 block">
            python -c "from ml.datasets.collector import DatasetCollector; print(DatasetCollector().audit_raw_data())"
          </code>
        </div>
      </div>
    </div>
  );
}
