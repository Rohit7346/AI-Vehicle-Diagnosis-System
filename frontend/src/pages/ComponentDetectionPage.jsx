import React, { useState, useEffect, useCallback } from 'react';
import {
  Layers,
  Cpu,
  AlertTriangle,
  AlertCircle,
  RefreshCw,
  CheckCircle2,
  ShieldCheck,
  RotateCcw,
} from 'lucide-react';
import ComponentImageUploader from '../components/ComponentImageUploader';
import DetectionViewer from '../components/DetectionViewer';
import ComponentDetectionResults from '../components/ComponentDetectionResults';
import apiService from '../services/api';

export default function ComponentDetectionPage({ systemHealth }) {
  // Model and Readiness State
  const [detectorStatus, setDetectorStatus] = useState(null);
  const [isStatusLoading, setIsStatusLoading] = useState(false);

  // File and Image State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageDimensions, setImageDimensions] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);

  // Inference Lifecycle State ('idle' | 'processing' | 'success' | 'error')
  const [uiState, setUiState] = useState('idle');
  const [detectionResult, setDetectionResult] = useState(null);
  const [errorNotice, setErrorNotice] = useState(null);

  // Interactive Hover/Selection State
  const [hoveredComponentId, setHoveredComponentId] = useState(null);
  const [selectedComponentId, setSelectedComponentId] = useState(null);

  // Fetch Component Detector Metadata
  const fetchStatus = useCallback(async () => {
    setIsStatusLoading(true);
    const res = await apiService.getComponentDetectorStatus();
    if (res.success) {
      setDetectorStatus(res.data);
    }
    setIsStatusLoading(false);
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Handle File Selection and extract natural image dimensions
  const handleFileSelect = (file) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setErrorNotice(null);
    setDetectionResult(null);
    setUiState('idle');

    // Read natural image dimensions
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;
  };

  // Clear / Reset Image
  const handleClearImage = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageDimensions(null);
    setDetectionResult(null);
    setErrorNotice(null);
    setUiState('idle');
    setHoveredComponentId(null);
    setSelectedComponentId(null);
  };

  // Run YOLOv8 Component Detection
  const handleRunDetection = async () => {
    if (!selectedFile) return;

    setUiState('processing');
    setErrorNotice(null);

    const res = await apiService.detectComponents(selectedFile, confidenceThreshold);

    if (res.success) {
      setDetectionResult(res.data);
      setUiState('success');
    } else {
      setErrorNotice({
        status: res.status,
        error: res.error,
        actionRequired: res.actionRequired,
        details: res.details,
      });
      setUiState('error');
    }
  };

  // Retry Action
  const handleRetry = () => {
    handleRunDetection();
  };

  const isModelLoaded = detectorStatus?.loaded ?? systemHealth?.models?.component_detector?.loaded ?? true;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
              <Layers className="w-5 h-5 text-blue-400" />
              <span>Vehicle Component Localization</span>
            </h2>
            <span className="px-2 py-0.5 bg-blue-950/60 border border-blue-900 text-blue-300 font-mono text-[11px] rounded">
              YOLOv8 Nano
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl">
            Locate key automotive subassemblies (engine bay, 12V battery, and radiator assembly) using the trained 
            three-class object detection model.
          </p>
        </div>

        {/* Model Status Badge */}
        <div className="flex items-center space-x-2 shrink-0">
          <button
            onClick={fetchStatus}
            disabled={isStatusLoading}
            className="p-2 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 rounded-lg transition"
            title="Refresh component detector status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isStatusLoading ? 'animate-spin' : ''}`} />
          </button>
          <div className="px-3 py-1.5 bg-slate-900 border border-slate-800 rounded-lg flex items-center space-x-2 text-xs">
            <Cpu className="w-4 h-4 text-slate-400" />
            <span className="text-slate-300 font-medium">Detector:</span>
            {isModelLoaded ? (
              <span className="flex items-center space-x-1 text-emerald-400 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" />
                <span>Active & Loaded</span>
              </span>
            ) : (
              <span className="flex items-center space-x-1 text-amber-400 font-semibold">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Weights Missing (503)</span>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Model Weights Missing Alert Notice */}
      {!isModelLoaded && (
        <div className="p-4 bg-amber-950/30 border border-amber-800/80 rounded-xl flex items-start space-x-3 text-xs text-amber-200">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-amber-100 text-sm">Model Weights Checkpoint Missing</h4>
            <p className="text-amber-300">
              The checkpoint <code className="text-amber-200 font-mono">ml/weights/component_detector_best.pt</code> was not detected 
              by the backend service. Inference requests will return HTTP 503 until valid weights are present.
            </p>
          </div>
        </div>
      )}

      {/* API Error / Service Unavailable Notice */}
      {errorNotice && (
        <div className="p-4 bg-rose-950/40 border border-rose-800/80 rounded-xl flex items-start space-x-3 text-xs text-rose-300 animate-fadeIn">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1 flex-1">
            <div className="flex items-center justify-between">
              <h4 className="font-bold text-rose-200 text-sm">
                Inference Request Failed (HTTP {errorNotice.status || 'Network Error'})
              </h4>
              <button
                onClick={handleRetry}
                className="px-2.5 py-1 bg-rose-900/60 hover:bg-rose-800 text-rose-200 border border-rose-700 rounded text-xs flex items-center space-x-1 transition font-medium"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Retry</span>
              </button>
            </div>
            <p className="text-rose-300 font-mono text-[11px] mt-1">{errorNotice.error}</p>
            {errorNotice.actionRequired && (
              <div className="mt-2 p-2.5 bg-rose-900/30 rounded border border-rose-800 text-[11px] text-rose-200">
                <span className="font-bold text-rose-100">Recommended Action: </span>
                {errorNotice.actionRequired}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Image Uploader & Controls */}
        <div className="lg:col-span-5 space-y-6">
          <ComponentImageUploader
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            imageDimensions={imageDimensions}
            confidenceThreshold={confidenceThreshold}
            onConfidenceChange={setConfidenceThreshold}
            onFileSelect={handleFileSelect}
            onClearImage={handleClearImage}
            onRunDetection={handleRunDetection}
            isLoading={uiState === 'processing'}
            detectorReady={isModelLoaded}
          />

          {/* Model Policy & Class Legend Card */}
          <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl text-xs space-y-3">
            <h4 className="font-semibold text-slate-300 flex items-center space-x-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Ontology & Class Operational Status</span>
            </h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2 bg-slate-950/60 rounded border border-slate-800/80">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
                  <span className="font-medium text-slate-200">Class 0: Engine Bay</span>
                </div>
                <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded border border-emerald-800 font-semibold">
                  Enabled
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-950/60 rounded border border-slate-800/80">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                  <span className="font-medium text-slate-200">Class 1: Battery</span>
                </div>
                <span className="text-[10px] bg-emerald-950 text-emerald-400 px-2 py-0.5 rounded border border-emerald-800 font-semibold">
                  Enabled
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-slate-950/60 rounded border border-slate-800/80">
                <div className="flex items-center space-x-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                  <span className="font-medium text-slate-200">Class 2: Radiator</span>
                </div>
                <span className="text-[10px] bg-amber-950 text-amber-300 px-2 py-0.5 rounded border border-amber-800 font-bold uppercase tracking-wider">
                  Experimental
                </span>
              </div>
            </div>
            <p className="text-[11px] text-slate-400">
              The radiator class is flagged as experimental due to limited exposed-core training representations.
              All radiator detections must be confirmed by a human technician.
            </p>
          </div>
        </div>

        {/* Right Column: Visualization & Results */}
        <div className="lg:col-span-7 space-y-6">
          {uiState === 'success' && detectionResult && previewUrl ? (
            <>
              {/* Interactive Bounding Box Visualization */}
              <DetectionViewer
                imageUrl={previewUrl}
                imageDimensions={imageDimensions || detectionResult.image}
                detections={detectionResult.detections}
                hoveredId={hoveredComponentId}
                onHoverDetection={setHoveredComponentId}
                selectedId={selectedComponentId}
                onSelectDetection={setSelectedComponentId}
              />

              {/* Structured Detection Results Panel */}
              <ComponentDetectionResults
                result={detectionResult}
                hoveredId={hoveredComponentId}
                onHoverDetection={setHoveredComponentId}
                selectedId={selectedComponentId}
                onSelectDetection={setSelectedComponentId}
              />
            </>
          ) : previewUrl ? (
            /* Image loaded but inference not yet executed */
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 text-center space-y-4">
              <h4 className="text-sm font-semibold text-slate-200">Image Staged for Analysis</h4>
              <div className="relative inline-block max-h-96 rounded-lg overflow-hidden border border-slate-700 shadow-xl">
                <img
                  src={previewUrl}
                  alt="Staged inspection"
                  className="max-h-96 object-contain block mx-auto"
                />
              </div>
              <p className="text-xs text-slate-400 max-w-md mx-auto">
                Click <span className="text-blue-400 font-semibold">"Detect Vehicle Components"</span> on the left
                to execute YOLOv8 inference and render interactive bounding box overlays.
              </p>
            </div>
          ) : (
            /* Empty State */
            <div className="bg-slate-900/40 border border-dashed border-slate-800 rounded-xl p-16 text-center text-slate-500 space-y-3">
              <div className="p-3 bg-slate-800 text-slate-400 rounded-full w-12 h-12 flex items-center justify-center mx-auto">
                <Layers className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-medium text-slate-300">No Image Selected</h4>
              <p className="text-xs text-slate-400 max-w-sm mx-auto">
                Upload a vehicle photograph on the left to localize engine bay, battery, and radiator components in real time.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
