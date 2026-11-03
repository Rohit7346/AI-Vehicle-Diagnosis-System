import React from 'react';

export default function FaultBadge({ severity }) {
  const styles = {
    CRITICAL: 'bg-rose-950/80 text-rose-300 border-rose-800/80',
    HIGH: 'bg-orange-950/80 text-orange-300 border-orange-800/80',
    MODERATE: 'bg-amber-950/80 text-amber-300 border-amber-800/80',
    LOW: 'bg-blue-950/80 text-blue-300 border-blue-800/80',
    UNKNOWN: 'bg-slate-800 text-slate-300 border-slate-700',
  };

  const currentStyle = styles[severity] || styles.UNKNOWN;

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide border uppercase ${currentStyle}`}>
      {severity}
    </span>
  );
}
