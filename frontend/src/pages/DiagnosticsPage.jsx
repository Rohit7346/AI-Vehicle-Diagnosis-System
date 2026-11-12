import React, { useState } from 'react';
import { AlertCircle, Info, ArrowRight, Layers, CheckCircle2 } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import DiagnosisResultCard from '../components/DiagnosisResultCard';
import apiService from '../services/api';

export default function DiagnosticsPage({ systemHealth, setActiveTab }) {
  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorNotice, setErrorNotice] = useState(null);

  const compLoaded = systemHealth?.models?.component_detector?.loaded;
  const faultLoaded = systemHealth?.models?.fault_detector?.loaded;
  const pipelineReady = compLoaded && faultLoaded;

  const handleDiagnose = async (file, confidence) => {
    setIsLoading(true);
    setErrorNotice(null);
    setReport(null);

    const result = await apiService.uploadDiagnosisImage(file, confidence);
    setIsLoading(false);

    if (result.success) {
      setReport(result.data);
    } else {
      setErrorNotice({
        status: result.status,
        error: result.error,
        details: result.details,
      });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Subassembly Diagnostic Console</h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload component photos to evaluate surface wear, corrosion, and leaks across localized vehicle subsystems.
        </p>
      </div>

      {/* Component Detector Available Notice */}
      {compLoaded && !faultLoaded && (
        <div className="p-4 bg-blue-950/30 border border-blue-800/80 rounded-xl flex items-start justify-between space-x-3 text-xs text-blue-200">
          <div className="flex items-start space-x-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="font-bold text-white text-sm">
                Stage 3 Component Detector is Live & Ready!
              </h4>
              <p className="text-slate-300">
                The 3-class YOLOv8 component detector (<code className="text-blue-300 font-mono">component_detector_best.pt</code>) is active. 
                However, this tab runs the <strong>dual-stage fault classification pipeline</strong>, which requires the visible defect detector (<code className="text-amber-300 font-mono">fault_detector_best.pt</code>) scheduled for later stages.
              </p>
            </div>
          </div>
          {setActiveTab && (
            <button
              onClick={() => setActiveTab('components')}
              className="px-3.5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition shrink-0 shadow-md shadow-blue-600/30"
            >
              <Layers className="w-4 h-4" />
              <span>Go to Component Detection</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}

      {/* Error / Service Unavailable Notice */}
      {errorNotice && (
        <div className="p-4 bg-rose-950/40 border border-rose-800/80 rounded-xl flex items-start space-x-3 text-xs text-rose-300">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-2 flex-1">
            <h4 className="font-bold text-rose-200 text-sm">
              Dual-Stage Inference Request Result: HTTP {errorNotice.status || 'Error'}
            </h4>
            <p className="text-rose-300 font-mono text-[11px]">
              {typeof errorNotice.error === 'string' ? errorNotice.error : JSON.stringify(errorNotice.error)}
            </p>
            {errorNotice.details?.action_required && (
              <div className="p-2.5 bg-rose-900/40 rounded border border-rose-800 text-[11px] text-rose-200">
                <span className="font-bold">Reason: </span>
                {errorNotice.details.action_required}
              </div>
            )}
            {compLoaded && setActiveTab && (
              <div className="pt-2">
                <button
                  onClick={() => setActiveTab('components')}
                  className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-md text-xs font-semibold flex items-center space-x-1.5 transition"
                >
                  <Layers className="w-3.5 h-3.5" />
                  <span>Open Component Detection Tab (Active Model)</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Left Column: Image Uploader */}
        <ImageUploader onDiagnose={handleDiagnose} isLoading={isLoading} />

        {/* Right Column: Diagnostic Results */}
        <div>
          {report ? (
            <DiagnosisResultCard report={report} />
          ) : (
            <div className="bg-slate-800/20 border border-dashed border-slate-800 rounded-xl p-12 text-center text-slate-500">
              <p className="text-sm font-medium text-slate-400">No active diagnostic report</p>
              <p className="text-xs text-slate-600 mt-1">
                Upload a subassembly image on the left to review detection bounding boxes, fault classifications, and corrective actions.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
