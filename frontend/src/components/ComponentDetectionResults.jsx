import React from 'react';
import {
  Cpu,
  Layers,
  Zap,
  Flame,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Info,
} from 'lucide-react';

const CLASS_CONFIG = {
  engine_bay: {
    title: 'Engine Bay',
    icon: Layers,
    color: 'text-cyan-400',
    border: 'border-cyan-500/40',
    bg: 'bg-cyan-950/30',
    pill: 'bg-cyan-900/60 text-cyan-300 border-cyan-700',
  },
  battery: {
    title: '12V Battery',
    icon: Zap,
    color: 'text-emerald-400',
    border: 'border-emerald-500/40',
    bg: 'bg-emerald-950/30',
    pill: 'bg-emerald-900/60 text-emerald-300 border-emerald-700',
  },
  radiator: {
    title: 'Radiator Core',
    icon: Flame,
    color: 'text-amber-400',
    border: 'border-amber-500/40',
    bg: 'bg-amber-950/30',
    pill: 'bg-amber-900/60 text-amber-300 border-amber-700',
  },
};

export default function ComponentDetectionResults({
  result,
  hoveredId = null,
  onHoverDetection = () => {},
  selectedId = null,
  onSelectDetection = () => {},
}) {
  if (!result) return null;

  const {
    model_name,
    model_version,
    device,
    inference_latency_ms,
    detections = [],
    summary = {},
    disclaimer,
  } = result;

  const totalDetections = summary.total_detections || 0;
  const engineBayCount = summary.class_counts?.engine_bay || 0;
  const batteryCount = summary.class_counts?.battery || 0;
  const radiatorCount = summary.class_counts?.radiator || 0;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm space-y-6">
      {/* Header and Model Metadata Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-4 border-b border-slate-800 gap-3">
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-mono text-xs text-blue-400 bg-blue-950/60 px-2.5 py-0.5 rounded border border-blue-900 flex items-center space-x-1">
              <Cpu className="w-3.5 h-3.5" />
              <span>{model_name}</span>
            </span>
            <span className="text-[11px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
              v{model_version}
            </span>
          </div>
          <h3 className="text-base font-bold text-slate-100 mt-1.5 flex items-center space-x-2">
            <span>Component Localization Report</span>
          </h3>
          <p className="text-xs text-slate-400">
            Automotive subsystem spatial boundary identification
          </p>
        </div>

        {/* Runtime Performance Badge */}
        <div className="flex items-center space-x-3 bg-slate-950/80 px-3.5 py-2 rounded-xl border border-slate-800 shrink-0">
          <div className="text-right">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Inference Latency</span>
            <div className="flex items-center space-x-1.5 justify-end">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              <span className="font-mono text-sm font-bold text-white">
                {inference_latency_ms.toFixed(1)} ms
              </span>
            </div>
          </div>
          <div className="h-6 w-px bg-slate-800" />
          <div className="text-left">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Compute Device</span>
            <span className="font-mono text-xs text-blue-300 font-semibold uppercase">
              {device}
            </span>
          </div>
        </div>
      </div>

      {/* Summary Scorecards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Total Detected */}
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
          <span className="text-[11px] text-slate-400 block font-medium">Total Detections</span>
          <span className="text-xl font-black text-white mt-0.5 block">{totalDetections}</span>
          <span className="text-[10px] text-slate-500">Components localized</span>
        </div>

        {/* Engine Bay */}
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-400 font-medium">Engine Bay</span>
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
          </div>
          <span className="text-xl font-black text-cyan-300 mt-0.5 block">{engineBayCount}</span>
          <span className="text-[10px] text-slate-500">Compartment perimeter</span>
        </div>

        {/* Battery */}
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-400 font-medium">12V Battery</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
          </div>
          <span className="text-xl font-black text-emerald-300 mt-0.5 block">{batteryCount}</span>
          <span className="text-[10px] text-slate-500">Lead-acid / AGM unit</span>
        </div>

        {/* Radiator (Experimental) */}
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg relative overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-400 font-medium">Radiator</span>
            <span className="px-1 py-0.2 bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded text-[9px] font-bold uppercase">
              Exp
            </span>
          </div>
          <span className="text-xl font-black text-amber-300 mt-0.5 block">{radiatorCount}</span>
          <span className="text-[10px] text-amber-400/90 font-medium">Provisional detection</span>
        </div>
      </div>

      {/* Empty State */}
      {totalDetections === 0 && (
        <div className="p-6 bg-slate-950/60 border border-dashed border-slate-800 rounded-xl text-center space-y-2">
          <Info className="w-6 h-6 text-slate-500 mx-auto" />
          <h4 className="text-sm font-semibold text-slate-300">No Target Components Identified</h4>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            The model did not localize an engine bay, battery, or radiator exceeding the confidence cutoff.
            Try lowering the confidence threshold or providing a closer, well-lit photo of the vehicle compartment.
          </p>
        </div>
      )}

      {/* Detections List */}
      {totalDetections > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
              <Layers className="w-4 h-4 text-blue-400" />
              <span>Identified Subsystem Instances ({totalDetections})</span>
            </h4>
            <span className="text-[11px] text-slate-500">
              Hover to locate &bull; Click to select
            </span>
          </div>

          <div className="space-y-2.5">
            {detections.map((det) => {
              const cfg = CLASS_CONFIG[det.class_name] || {
                title: det.class_name,
                icon: Layers,
                color: 'text-blue-400',
                border: 'border-slate-800',
                bg: 'bg-slate-900',
                pill: 'bg-slate-800 text-slate-300',
              };
              const Icon = cfg.icon;
              const isHovered = hoveredId === det.component_id;
              const isSelected = selectedId === det.component_id;

              return (
                <div
                  key={det.component_id}
                  onMouseEnter={() => onHoverDetection(det.component_id)}
                  onMouseLeave={() => onHoverDetection(null)}
                  onClick={() => onSelectDetection(det.component_id)}
                  className={`p-3.5 rounded-lg border transition-all cursor-pointer ${
                    isHovered || isSelected
                      ? 'bg-slate-850 border-blue-500/80 shadow-md ring-1 ring-blue-500/40'
                      : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center space-x-2.5">
                      <div className={`p-2 rounded-lg ${cfg.bg} border ${cfg.border}`}>
                        <Icon className={`w-4 h-4 ${cfg.color}`} />
                      </div>
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-bold text-white capitalize">
                            {cfg.title}
                          </span>
                          <span className="font-mono text-[10px] text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                            {det.component_id}
                          </span>
                          {det.is_experimental ? (
                            <span className="px-2 py-0.5 bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded text-[10px] font-bold uppercase tracking-wider flex items-center space-x-1">
                              <AlertTriangle className="w-3 h-3" />
                              <span>Experimental</span>
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 bg-emerald-950/60 text-emerald-400 border border-emerald-800/60 rounded text-[10px] font-semibold flex items-center space-x-1">
                              <CheckCircle2 className="w-3 h-3" />
                              <span>Validated</span>
                            </span>
                          )}
                        </div>

                        {/* Normalized & Pixel Coordinates */}
                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] font-mono text-slate-400">
                          <span>
                            Pixels: [{det.box_pixels.x_min}, {det.box_pixels.y_min}, {det.box_pixels.x_max}, {det.box_pixels.y_max}]
                          </span>
                          <span className="text-slate-600">&bull;</span>
                          <span>
                            Norm: [{det.box_normalized.x_min.toFixed(2)}, {det.box_normalized.y_min.toFixed(2)}, {det.box_normalized.x_max.toFixed(2)}, {det.box_normalized.y_max.toFixed(2)}]
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Confidence Meter */}
                    <div className="text-right shrink-0">
                      <span className="text-[10px] text-slate-400 block">Localization Conf.</span>
                      <span className="font-mono text-sm font-bold text-blue-400">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                      <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1 ml-auto">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${Math.min(100, Math.max(5, det.confidence * 100))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Operational Warning Callout if Present */}
                  {det.warning && (
                    <div className="mt-2.5 p-2 bg-amber-950/30 border border-amber-800/60 rounded text-[11px] text-amber-200/90 flex items-start space-x-2">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                      <div className="space-y-0.5">
                        <span className="font-semibold text-amber-300 block">Operational Policy Notice:</span>
                        <p>{det.warning}</p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Mandatory Ethical Disclaimer Box */}
      <div className="p-3.5 bg-blue-950/20 border border-blue-900/60 rounded-xl space-y-1.5 text-xs text-blue-200">
        <div className="flex items-center space-x-2 text-blue-300 font-semibold">
          <Info className="w-4 h-4 shrink-0 text-blue-400" />
          <span>Important Diagnostic Limitations & Ethical Boundaries</span>
        </div>
        <p className="text-slate-300 leading-relaxed text-[11px]">
          {disclaimer}
        </p>
        <div className="pt-1 border-t border-blue-900/40 text-[10px] text-slate-400 flex flex-wrap gap-x-4 gap-y-1">
          <span>&bull; Component presence &ne; component health</span>
          <span>&bull; Fault classification model is not yet available</span>
          <span>&bull; Radiator detections require manual confirmation</span>
        </div>
      </div>
    </div>
  );
}
