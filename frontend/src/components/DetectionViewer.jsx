import React, { useState, useRef, useMemo } from 'react';
import { Eye, EyeOff, Layers, AlertTriangle } from 'lucide-react';

const CLASS_THEMES = {
  engine_bay: {
    name: 'Engine Bay',
    stroke: '#06b6d4', // cyan-500
    fill: 'rgba(6, 182, 212, 0.16)',
    borderClass: 'border-cyan-500',
    bgBadge: 'bg-cyan-950/80 border-cyan-500/80 text-cyan-300',
    dotColor: 'bg-cyan-400',
    dashed: false,
  },
  battery: {
    name: '12V Battery',
    stroke: '#10b981', // emerald-500
    fill: 'rgba(16, 185, 129, 0.18)',
    borderClass: 'border-emerald-500',
    bgBadge: 'bg-emerald-950/80 border-emerald-500/80 text-emerald-300',
    dotColor: 'bg-emerald-400',
    dashed: false,
  },
  radiator: {
    name: 'Radiator Assembly',
    stroke: '#f59e0b', // amber-500
    fill: 'rgba(245, 158, 11, 0.20)',
    borderClass: 'border-amber-500',
    bgBadge: 'bg-amber-950/80 border-amber-500/80 text-amber-300',
    dotColor: 'bg-amber-400',
    dashed: true,
  },
};

const DEFAULT_THEME = {
  name: 'Component',
  stroke: '#3b82f6',
  fill: 'rgba(59, 130, 246, 0.15)',
  borderClass: 'border-blue-500',
  bgBadge: 'bg-blue-950/80 border-blue-500/80 text-blue-300',
  dotColor: 'bg-blue-400',
  dashed: false,
};

export default function DetectionViewer({
  imageUrl,
  imageDimensions,
  detections = [],
  hoveredId = null,
  onHoverDetection = () => {},
  selectedId = null,
  onSelectDetection = () => {},
}) {
  const [visibleClasses, setVisibleClasses] = useState({
    engine_bay: true,
    battery: true,
    radiator: true,
  });
  const [coordMode, setCoordMode] = useState('normalized'); // 'normalized' | 'pixels'
  const [showLabels, setShowLabels] = useState(true);
  const containerRef = useRef(null);

  const naturalWidth = imageDimensions?.width || 640;
  const naturalHeight = imageDimensions?.height || 480;

  // Filter detections based on toggle visibility
  const filteredDetections = useMemo(() => {
    return detections.filter((d) => visibleClasses[d.class_name] !== false);
  }, [detections, visibleClasses]);

  const toggleClassVisibility = (className) => {
    setVisibleClasses((prev) => ({
      ...prev,
      [className]: !prev[className],
    }));
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-lg flex flex-col">
      {/* Viewer Toolbar */}
      <div className="px-4 py-2.5 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-blue-400" />
          <span className="font-semibold text-slate-200">Interactive Visual Inspection</span>
          <span className="text-[11px] text-slate-400">
            ({filteredDetections.length} of {detections.length} visible)
          </span>
        </div>

        {/* Visibility Toggles */}
        <div className="flex items-center flex-wrap gap-2">
          {['engine_bay', 'battery', 'radiator'].map((clsName) => {
            const theme = CLASS_THEMES[clsName] || DEFAULT_THEME;
            const isVisible = visibleClasses[clsName];
            const count = detections.filter((d) => d.class_name === clsName).length;
            return (
              <button
                key={clsName}
                onClick={() => toggleClassVisibility(clsName)}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium transition border ${
                  isVisible
                    ? 'bg-slate-800 border-slate-700 text-slate-200'
                    : 'bg-slate-950/60 border-slate-800/80 text-slate-500 line-through'
                }`}
                title={`Toggle ${theme.name} overlays`}
              >
                <span className={`w-2 h-2 rounded-full ${theme.dotColor}`} />
                <span>{theme.name}</span>
                <span className="text-[10px] opacity-75 font-mono">({count})</span>
              </button>
            );
          })}

          <div className="h-4 w-px bg-slate-800 mx-1 hidden sm:block" />

          {/* Coordinate System Toggle */}
          <button
            onClick={() => setCoordMode((prev) => (prev === 'normalized' ? 'pixels' : 'normalized'))}
            className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md text-[11px] text-slate-300 font-mono transition"
            title="Switch coordinate format display"
          >
            Coords: {coordMode === 'normalized' ? '[0.0, 1.0]' : '[px]'}
          </button>

          {/* Toggle Label Tags */}
          <button
            onClick={() => setShowLabels((prev) => !prev)}
            className="p-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-md text-slate-300 transition"
            title={showLabels ? 'Hide label badges' : 'Show label badges'}
          >
            {showLabels ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Viewport Container */}
      <div className="relative w-full bg-slate-950 flex items-center justify-center p-3 overflow-hidden min-h-[380px]">
        <div
          ref={containerRef}
          className="relative inline-block max-w-full rounded-lg overflow-hidden border border-slate-800/80 shadow-2xl"
          style={{
            maxHeight: '620px',
            aspectRatio: `${naturalWidth} / ${naturalHeight}`,
          }}
        >
          {/* Base Image */}
          <img
            src={imageUrl}
            alt="Vehicle inspection subsystem"
            className="block w-full h-full object-contain select-none pointer-events-none"
          />

          {/* Scalable SVG Bounding Box Overlay */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            viewBox={`0 0 ${naturalWidth} ${naturalHeight}`}
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              <pattern
                id="experimental-stripe"
                width="10"
                height="10"
                patternTransform="rotate(45 0 0)"
                patternUnits="userSpaceOnUse"
              >
                <line x1="0" y1="0" x2="0" y2="10" stroke="#f59e0b" strokeWidth="2" strokeOpacity="0.3" />
              </pattern>
            </defs>

            {filteredDetections.map((det) => {
              const theme = CLASS_THEMES[det.class_name] || DEFAULT_THEME;
              const isHovered = hoveredId === det.component_id;
              const isSelected = selectedId === det.component_id;

              // Read coordinates (pixels conform directly to SVG viewBox)
              const x = det.box_pixels.x_min;
              const y = det.box_pixels.y_min;
              const w = Math.max(1, det.box_pixels.x_max - det.box_pixels.x_min);
              const h = Math.max(1, det.box_pixels.y_max - det.box_pixels.y_min);

              const strokeColor = theme.stroke;
              const strokeWidth = isHovered || isSelected ? 4 : 2.5;

              return (
                <g
                  key={det.component_id}
                  className="transition-all duration-150 cursor-pointer pointer-events-auto"
                  onMouseEnter={() => onHoverDetection(det.component_id)}
                  onMouseLeave={() => onHoverDetection(null)}
                  onClick={() => onSelectDetection(det.component_id)}
                >
                  {/* Bounding Rectangle Fill */}
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    fill={det.is_experimental ? 'url(#experimental-stripe)' : theme.fill}
                    className="transition-colors duration-150"
                  />

                  {/* Bounding Box Border */}
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={theme.dashed ? '8,6' : 'none'}
                    className={isHovered || isSelected ? 'filter drop-shadow(0 0 6px ' + strokeColor + ')' : ''}
                  />

                  {/* Corner Accent Grips */}
                  <circle cx={x} cy={y} r="3.5" fill={strokeColor} />
                  <circle cx={x + w} cy={y} r="3.5" fill={strokeColor} />
                  <circle cx={x} cy={y + h} r="3.5" fill={strokeColor} />
                  <circle cx={x + w} cy={y + h} r="3.5" fill={strokeColor} />
                </g>
              );
            })}
          </svg>

          {/* HTML Overlay for Rich CSS Badges & Tooltips */}
          {showLabels && (
            <div className="absolute inset-0 pointer-events-none">
              {filteredDetections.map((det) => {
                const theme = CLASS_THEMES[det.class_name] || DEFAULT_THEME;
                const isHovered = hoveredId === det.component_id;
                const isSelected = selectedId === det.component_id;

                // Position via normalized coordinates to maintain responsive alignment
                const leftPercent = det.box_normalized.x_min * 100;
                const topPercent = det.box_normalized.y_min * 100;

                const formattedCoords =
                  coordMode === 'normalized'
                    ? `[${det.box_normalized.x_min.toFixed(2)}, ${det.box_normalized.y_min.toFixed(2)}, ${det.box_normalized.x_max.toFixed(2)}, ${det.box_normalized.y_max.toFixed(2)}]`
                    : `[${det.box_pixels.x_min}, ${det.box_pixels.y_min}, ${det.box_pixels.x_max}, ${det.box_pixels.y_max}]`;

                return (
                  <div
                    key={det.component_id}
                    className="absolute pointer-events-auto transform -translate-y-full mb-1 transition-all"
                    style={{
                      left: `${leftPercent}%`,
                      top: `${topPercent}%`,
                      zIndex: isHovered || isSelected ? 30 : 20,
                    }}
                    onMouseEnter={() => onHoverDetection(det.component_id)}
                    onMouseLeave={() => onHoverDetection(null)}
                    onClick={() => onSelectDetection(det.component_id)}
                  >
                    <div
                      className={`inline-flex items-center space-x-1.5 px-2 py-0.5 rounded shadow-md border backdrop-blur-md text-[11px] font-semibold transition-all ${
                        theme.bgBadge
                      } ${isHovered || isSelected ? 'scale-105 ring-2 ring-white/60' : ''}`}
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${theme.dotColor}`} />
                      <span className="capitalize">{det.class_name.replace(/_/g, ' ')}</span>
                      <span className="font-mono text-[10px] opacity-90">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                      {det.is_experimental && (
                        <span className="bg-amber-500/20 text-amber-300 text-[9px] px-1 py-0.2 rounded border border-amber-500/40 uppercase tracking-wider font-bold">
                          Exp
                        </span>
                      )}
                    </div>

                    {/* Extended Info Tooltip on Hover */}
                    {isHovered && (
                      <div className="absolute left-0 top-full mt-1.5 w-64 p-2.5 bg-slate-900/95 border border-slate-700 rounded-lg shadow-2xl text-[11px] text-slate-200 z-50 pointer-events-none space-y-1 backdrop-blur-md">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-1">
                          <span className="font-bold text-white capitalize">
                            {det.class_name.replace(/_/g, ' ')} ({det.component_id})
                          </span>
                          <span className="font-mono text-blue-300 text-[10px]">
                            Conf: {(det.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-[10px] text-slate-400 font-mono">
                          Coords: {formattedCoords}
                        </div>
                        {det.warning && (
                          <div className="mt-1 p-1.5 bg-amber-950/40 border border-amber-800/80 rounded text-[10px] text-amber-300 flex items-start space-x-1">
                            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                            <span>{det.warning}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Viewer Footer Legend */}
      <div className="px-4 py-2 bg-slate-900/90 border-t border-slate-800 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
        <div className="flex items-center space-x-4">
          <span className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-cyan-500/30 border border-cyan-500" />
            <span>Engine Bay (Validated)</span>
          </span>
          <span className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/30 border border-emerald-500" />
            <span>Battery (Validated)</span>
          </span>
          <span className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-amber-500/30 border border-amber-500 border-dashed" />
            <span>Radiator (Experimental)</span>
          </span>
        </div>
        <div className="text-[10px] text-slate-500">
          Source Resolution: {naturalWidth} &times; {naturalHeight} px
        </div>
      </div>
    </div>
  );
}
