import React from 'react';
import { Layers } from 'lucide-react';

export default function ComponentList({ components }) {
  if (!components || components.length === 0) {
    return (
      <div className="p-4 bg-slate-900/40 rounded-lg border border-slate-800 text-center text-xs text-slate-500">
        No automotive subsystems identified in this scan.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {components.map((comp, idx) => (
        <div
          key={idx}
          className="flex items-center justify-between p-3 bg-slate-900/60 rounded-lg border border-slate-800 text-xs"
        >
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-slate-200 capitalize">
              {comp.label.replace('_', ' ')}
            </span>
          </div>
          <div className="flex items-center space-x-3 text-slate-400">
            <span>Confidence: {(comp.confidence * 100).toFixed(1)}%</span>
            <span className="font-mono text-[10px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
              [{comp.bbox.join(', ')}]
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
