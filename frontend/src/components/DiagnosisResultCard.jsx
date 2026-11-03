import React from 'react';
import { AlertCircle, CheckCircle, ShieldAlert, Wrench, Layers } from 'lucide-react';
import FaultBadge from './FaultBadge';
import ComponentList from './ComponentList';

export default function DiagnosisResultCard({ report }) {
  if (!report) return null;

  const { summary, components, faults, image_metadata, inference_id, timestamp } = report;
  const score = summary.overall_health_score;

  const getScoreColor = (val) => {
    if (val >= 80) return 'text-emerald-400 border-emerald-500/40 bg-emerald-950/20';
    if (val >= 50) return 'text-amber-400 border-amber-500/40 bg-amber-950/20';
    return 'text-rose-400 border-rose-500/40 bg-rose-950/20';
  };

  return (
    <div className="bg-slate-800/40 border border-slate-800 rounded-xl p-5 shadow-sm space-y-6">
      {/* Top Banner & Health Score */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between pb-4 border-b border-slate-800 gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-mono text-xs text-blue-400 bg-blue-950/60 px-2 py-0.5 rounded border border-blue-900">
              ID: {inference_id}
            </span>
            <span className="text-xs text-slate-400">
              {new Date(timestamp).toLocaleTimeString()}
            </span>
          </div>
          <h3 className="text-base font-bold text-slate-100 mt-1">
            Diagnostic Inspection Report
          </h3>
          <p className="text-xs text-slate-400">{summary.status_summary}</p>
        </div>

        {/* Dynamic Health Score Gauge */}
        <div className={`flex items-center space-x-3 px-4 py-2.5 rounded-xl border ${getScoreColor(score)}`}>
          <div>
            <span className="text-[10px] uppercase font-bold tracking-wider block">Subassembly Health</span>
            <span className="text-2xl font-black">{score} / 100</span>
          </div>
        </div>
      </div>

      {/* Detected Defects & Prescriptive Remediation */}
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-rose-400" />
          <span>Localised Physical Defects ({faults.length})</span>
        </h4>

        {faults.length === 0 ? (
          <div className="p-4 bg-emerald-950/20 border border-emerald-800/40 rounded-lg flex items-center space-x-3 text-xs text-emerald-300">
            <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0" />
            <span>No visible cracks, leaks, corrosion, or severe wear detected within inspection parameters.</span>
          </div>
        ) : (
          <div className="space-y-3">
            {faults.map((fault, idx) => (
              <div
                key={idx}
                className="p-3.5 bg-slate-900/60 border border-slate-800 rounded-lg space-y-2 text-xs"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-slate-200 capitalize text-sm">
                      {fault.label.replace(/_/g, ' ')}
                    </span>
                    <FaultBadge severity={fault.severity} />
                  </div>
                  <span className="text-slate-400 text-xs">
                    Confidence: {(fault.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                {fault.associated_component && (
                  <div className="text-[11px] text-slate-400 flex items-center space-x-1">
                    <span>Host Subsystem:</span>
                    <span className="text-blue-300 font-medium capitalize">
                      {fault.associated_component.replace(/_/g, ' ')}
                    </span>
                  </div>
                )}

                <div className="p-2.5 bg-slate-950/70 border border-slate-800/80 rounded flex items-start space-x-2 text-slate-300">
                  <Wrench className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold text-slate-200 block text-[11px]">Recommended Engineering Action:</span>
                    <span>{fault.recommendation}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Identified Subsystems */}
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center space-x-2">
          <Layers className="w-4 h-4 text-blue-400" />
          <span>Subsystems Identified in Inspection Field ({components.length})</span>
        </h4>
        <ComponentList components={components} />
      </div>

      {/* Image Metadata Bar */}
      <div className="pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
        <span>Image Dimensions: {image_metadata.original_width} &times; {image_metadata.original_height} px</span>
        <span>CLAHE Filter Applied: {image_metadata.clahe_applied ? 'Yes' : 'No'}</span>
      </div>
    </div>
  );
}
