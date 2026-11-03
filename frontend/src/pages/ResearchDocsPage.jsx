import React from 'react';
import { BookOpen, FileText, CheckCircle2, ShieldCheck, Layers, GitBranch } from 'lucide-react';

export default function ResearchDocsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">Research Specifications & Methodology</h2>
        <p className="text-xs text-slate-400 mt-1">
          Academic project architecture, multimodal design taxonomy, and documentation indices.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Architecture Spec */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2">
          <div className="flex items-center space-x-2 text-blue-400">
            <Layers className="w-4 h-4" />
            <h3 className="text-sm font-bold text-white">System Architecture Document</h3>
          </div>
          <p className="text-xs text-slate-400">
            Examines the two-stage visual perception hierarchy, FastAPI gateway patterns, and clean separation between model weights and UI.
          </p>
          <div className="pt-2 text-[11px] font-mono text-slate-500">
            Path: <span className="text-slate-300">docs/architecture.md</span>
          </div>
        </div>

        {/* Dataset Spec */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2">
          <div className="flex items-center space-x-2 text-emerald-400">
            <FileText className="w-4 h-4" />
            <h3 className="text-sm font-bold text-white">Dataset & Collection Protocols</h3>
          </div>
          <p className="text-xs text-slate-400">
            Defines photographic standards (1080p+, illumination, angle ranges) and the YOLO annotation format for the 10 subsystems and 12 defect classes.
          </p>
          <div className="pt-2 text-[11px] font-mono text-slate-500">
            Path: <span className="text-slate-300">docs/dataset_specifications.md</span>
          </div>
        </div>

        {/* Methodology Spec */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2">
          <div className="flex items-center space-x-2 text-purple-400">
            <GitBranch className="w-4 h-4" />
            <h3 className="text-sm font-bold text-white">Model Methodology & Loss Formulation</h3>
          </div>
          <p className="text-xs text-slate-400">
            Outlines the mathematical formulations for component detection, Intersection-over-Area (IoA) fault attribution, and health score calculations.
          </p>
          <div className="pt-2 text-[11px] font-mono text-slate-500">
            Path: <span className="text-slate-300">docs/model_methodology.md</span>
          </div>
        </div>

        {/* API Spec */}
        <div className="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2">
          <div className="flex items-center space-x-2 text-amber-400">
            <BookOpen className="w-4 h-4" />
            <h3 className="text-sm font-bold text-white">REST API Specifications</h3>
          </div>
          <p className="text-xs text-slate-400">
            Lists the OpenAPI schema contracts, health probe structures, image multipart payload constraints, and HTTP status codes.
          </p>
          <div className="pt-2 text-[11px] font-mono text-slate-500">
            Path: <span className="text-slate-300">docs/api_specifications.md</span>
          </div>
        </div>
      </div>

      {/* Engineering Project Ethics & Safeguards */}
      <div className="p-5 bg-slate-800/30 border border-slate-800 rounded-xl space-y-3">
        <h3 className="text-sm font-bold text-slate-200 flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Engineering Integrity & Research Safeguards</span>
        </h3>
        <ul className="text-xs text-slate-400 space-y-2">
          <li className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-slate-200">Zero Simulated/Mock AI:</strong> If model weights are missing, the system reports uninitialized status instead of generating fake bounding boxes.</span>
          </li>
          <li className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-slate-200">Automated Pre-Flight Gate:</strong> Training scripts halt immediately if fewer than the minimum threshold of valid images or labels exist.</span>
          </li>
          <li className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span><strong className="text-slate-200">Independent Decoupled Tiers:</strong> ML logic can be run via CLI without FastAPI or React running, and backend can be tested independently.</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
