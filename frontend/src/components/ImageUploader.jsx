import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sliders, AlertCircle, Loader2 } from 'lucide-react';

export default function ImageUploader({ onDiagnose, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [confidence, setConfidence] = useState(0.35);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    onDiagnose(selectedFile, confidence);
  };

  return (
    <div className="bg-slate-800/40 border border-slate-800 rounded-xl p-5 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center space-x-2">
        <UploadCloud className="w-4 h-4 text-blue-400" />
        <span>Vehicle Inspection Image Ingestion</span>
      </h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Drag and Drop Zone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-xl p-6 text-center cursor-pointer transition bg-slate-900/40 hover:bg-slate-900/80"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
          />

          {previewUrl ? (
            <div className="flex flex-col items-center">
              <img
                src={previewUrl}
                alt="Selected subassembly preview"
                className="max-h-56 rounded-lg object-contain border border-slate-700 shadow-md mb-2"
              />
              <span className="text-xs text-slate-300 font-medium">
                {selectedFile?.name} ({((selectedFile?.size || 0) / (1024 * 1024)).toFixed(2)} MB)
              </span>
              <span className="text-[11px] text-blue-400 mt-1">Click or drag another image to replace</span>
            </div>
          ) : (
            <div className="flex flex-col items-center py-4">
              <div className="p-3 bg-slate-800 text-slate-400 rounded-full mb-3">
                <ImageIcon className="w-8 h-8" />
              </div>
              <p className="text-sm font-medium text-slate-200">
                Click to upload or drag & drop inspection photo
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supports High-Res JPEG, PNG, WEBP (Max 25MB)
              </p>
            </div>
          )}
        </div>

        {/* Confidence Threshold Parameter */}
        <div className="flex items-center justify-between p-3 bg-slate-900/60 rounded-lg border border-slate-800">
          <div className="flex items-center space-x-2">
            <Sliders className="w-4 h-4 text-slate-400" />
            <label className="text-xs text-slate-300 font-medium">
              Confidence Filter: <span className="font-mono text-blue-400">{(confidence * 100).toFixed(0)}%</span>
            </label>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-48 accent-blue-500 cursor-pointer"
          />
        </div>

        {/* Action Button */}
        <button
          type="submit"
          disabled={!selectedFile || isLoading}
          className={`w-full py-2.5 px-4 rounded-lg text-sm font-semibold flex items-center justify-center space-x-2 transition ${
            !selectedFile || isLoading
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/30'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Executing Dual-Stage Perception Pipeline...</span>
            </>
          ) : (
            <>
              <UploadCloud className="w-4 h-4" />
              <span>Analyze Vehicle Subsystem</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
