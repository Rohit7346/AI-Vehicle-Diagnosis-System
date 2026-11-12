import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  Image as ImageIcon,
  Sliders,
  AlertCircle,
  Loader2,
  Trash2,
  RefreshCw,
  FileCheck,
} from 'lucide-react';

const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp'];

export default function ComponentImageUploader({
  selectedFile,
  previewUrl,
  imageDimensions,
  confidenceThreshold,
  onConfidenceChange,
  onFileSelect,
  onClearImage,
  onRunDetection,
  isLoading,
  detectorReady = true,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);
  const fileInputRef = useRef(null);

  const validateAndProcessFile = (file) => {
    setValidationError(null);

    if (!file) return false;

    // Check MIME type or filename extension
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    const isMimeValid = ALLOWED_MIME_TYPES.includes(file.type);
    const isExtValid = ALLOWED_EXTENSIONS.includes(extension);

    if (!isMimeValid && !isExtValid) {
      setValidationError(
        `Unsupported file type "${file.type || extension}". Supported formats: JPEG (.jpg, .jpeg), PNG (.png), and WEBP (.webp).`
      );
      return false;
    }

    // Check File Size
    if (file.size > MAX_FILE_SIZE_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      setValidationError(
        `File size (${sizeMb} MB) exceeds maximum permitted limit of 25 MB. Please select a smaller or compressed image.`
      );
      return false;
    }

    if (file.size === 0) {
      setValidationError('The selected file is empty (0 bytes). Please upload a valid vehicle photograph.');
      return false;
    }

    onFileSelect(file);
    return true;
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
    // Reset input value so re-selecting same file triggers change
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isLoading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isLoading) return;

    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile || isLoading) return;
    onRunDetection();
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
          <UploadCloud className="w-4 h-4 text-blue-400" />
          <span>Vehicle Inspection Image Ingestion</span>
        </h3>
        {selectedFile && !isLoading && (
          <button
            type="button"
            onClick={onClearImage}
            className="flex items-center space-x-1 text-xs text-rose-400 hover:text-rose-300 transition"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Remove Image</span>
          </button>
        )}
      </div>

      {/* Validation Error Banner */}
      {validationError && (
        <div className="p-3 bg-rose-950/40 border border-rose-800/80 rounded-lg flex items-start space-x-2 text-xs text-rose-300 animate-fadeIn">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold block">File Validation Notice</span>
            <p>{validationError}</p>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Drag and Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => {
            if (!isLoading) fileInputRef.current?.click();
          }}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition relative overflow-hidden ${
            isDragging
              ? 'border-blue-500 bg-blue-950/20 shadow-lg shadow-blue-500/10'
              : selectedFile
              ? 'border-slate-700 bg-slate-950/60 hover:border-slate-600'
              : 'border-slate-700 hover:border-blue-500 bg-slate-950/40 hover:bg-slate-950/80'
          } ${isLoading ? 'opacity-60 cursor-not-allowed' : ''}`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp"
            disabled={isLoading}
            className="hidden"
          />

          {previewUrl ? (
            <div className="flex flex-col items-center">
              <div className="relative group mb-3">
                <img
                  src={previewUrl}
                  alt="Selected inspection preview"
                  className="max-h-56 rounded-lg object-contain border border-slate-700 shadow-md transition group-hover:opacity-90"
                />
                <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 flex items-center justify-center rounded-lg transition">
                  <span className="text-xs font-semibold text-white bg-slate-900/90 px-3 py-1.5 rounded-md border border-slate-700 flex items-center space-x-1.5 shadow">
                    <RefreshCw className="w-3.5 h-3.5 text-blue-400" />
                    <span>Click to Replace File</span>
                  </span>
                </div>
              </div>

              <div className="flex items-center space-x-2 text-xs text-slate-300 font-medium">
                <FileCheck className="w-4 h-4 text-emerald-400" />
                <span className="truncate max-w-xs">{selectedFile.name}</span>
                <span className="text-slate-500 font-mono">
                  ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                </span>
                {imageDimensions && (
                  <span className="text-[11px] text-blue-400 font-mono">
                    [{imageDimensions.width}&times;{imageDimensions.height}]
                  </span>
                )}
              </div>
              <span className="text-[11px] text-slate-400 mt-1">
                Drag a replacement image here or click to browse
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center py-6">
              <div className="p-3.5 bg-slate-800 text-blue-400 rounded-full mb-3 shadow-inner">
                <ImageIcon className="w-8 h-8" />
              </div>
              <p className="text-sm font-medium text-slate-200">
                Click to browse or drag & drop vehicle photo
              </p>
              <p className="text-xs text-slate-400 mt-1 max-w-sm">
                Supported formats: High-Resolution JPEG, PNG, WEBP (Max file size: 25 MB)
              </p>
              <div className="mt-3 flex items-center space-x-2 text-[11px] text-slate-500">
                <span className="px-2 py-0.5 bg-slate-800/80 rounded border border-slate-700/60 font-mono">.JPG</span>
                <span className="px-2 py-0.5 bg-slate-800/80 rounded border border-slate-700/60 font-mono">.PNG</span>
                <span className="px-2 py-0.5 bg-slate-800/80 rounded border border-slate-700/60 font-mono">.WEBP</span>
              </div>
            </div>
          )}
        </div>

        {/* Confidence Threshold Parameter Slider */}
        <div className="p-3.5 bg-slate-950/60 rounded-lg border border-slate-800 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2 text-slate-300">
              <Sliders className="w-4 h-4 text-slate-400" />
              <span className="font-medium">Detection Confidence Cutoff:</span>
            </div>
            <span className="font-mono text-blue-400 font-bold bg-blue-950/60 px-2 py-0.5 rounded border border-blue-900">
              {(confidenceThreshold * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <span className="text-[10px] text-slate-400 font-mono">10%</span>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => onConfidenceChange(parseFloat(e.target.value))}
              disabled={isLoading}
              className="flex-1 accent-blue-500 cursor-pointer disabled:cursor-not-allowed"
            />
            <span className="text-[10px] text-slate-400 font-mono">90%</span>
          </div>
          <p className="text-[11px] text-slate-400">
            Recommended baseline is 25%. Lower values capture fainter detections; higher values reduce false positives.
          </p>
        </div>

        {/* Primary Inference Action Button */}
        <button
          type="submit"
          disabled={!selectedFile || isLoading || !detectorReady}
          className={`w-full py-3 px-4 rounded-lg text-sm font-semibold flex items-center justify-center space-x-2 transition ${
            !selectedFile || isLoading || !detectorReady
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/30 active:scale-[0.99]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin text-white" />
              <span>Executing YOLOv8 Subsystem Localization...</span>
            </>
          ) : (
            <>
              <UploadCloud className="w-4 h-4" />
              <span>Detect Vehicle Components</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
}
