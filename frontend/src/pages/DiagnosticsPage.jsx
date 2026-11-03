import React, { useState } from 'react';
import { AlertTriangle, AlertCircle, Info, CheckCircle2 } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import DiagnosisResultCard from '../components/DiagnosisResultCard';
import apiService from '../services/api';

export default function DiagnosticsPage({ systemHealth }) {
  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorNotice, setErrorNotice] = useState(null);

  const pipelineReady = systemHealth?.models?.pipeline_ready;

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
          Upload component photos to localize mechanical subsystems and evaluate surface wear, corrosion, and leaks.
        </p>
      </div>

      {/* Weights Missing Guidance Alert */}
      {!pipelineReady && (
        <div className="p-4 bg-slate-900 border border-slate-700/80 rounded-xl flex items-start space-x-3 text-xs text-slate-300">
          <Info className="w-5 h-5 text-blue-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-semibold text-slate-200 text-sm">Truthful Research Mode Active</h4>
            <p className="text-slate-400">
              Model weight checkpoints (<code className="text-blue-300">component_detector_best.pt</code>, <code className="text-blue-300">fault_detector_best.pt</code>) have not yet been produced. 
              The system strictly avoids emitting fabricated bounding boxes. Submitting an image will query the API and return the verified status.
            </p>
          </div>
        </div>
      )}

      {/* Error / Service Unavailable Notice */}
      {errorNotice && (
        <div className="p-4 bg-rose-950/40 border border-rose-800/80 rounded-xl flex items-start space-x-3 text-xs text-rose-300">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-bold text-rose-200 text-sm">
              Inference Request Result: HTTP {errorNotice.status || 'Error'}
            </h4>
            <p className="text-rose-300 font-mono text-[11px]">
              {typeof errorNotice.error === 'string' ? errorNotice.error : JSON.stringify(errorNotice.error)}
            </p>
            {errorNotice.details?.action_required && (
              <div className="mt-2 p-2 bg-rose-900/40 rounded border border-rose-800 text-[11px] text-rose-200">
                <span className="font-bold">Action Required: </span>
                {errorNotice.details.action_required}
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
