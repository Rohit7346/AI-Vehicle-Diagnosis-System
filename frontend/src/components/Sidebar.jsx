import React from 'react';
import { LayoutDashboard, Stethoscope, Cpu, BookOpen, Layers } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'dashboard', label: 'Executive Dashboard', icon: LayoutDashboard },
    { id: 'diagnostics', label: 'Subsystem Diagnosis', icon: Stethoscope },
    { id: 'models', label: 'Model Architecture & Weights', icon: Cpu },
    { id: 'docs', label: 'Research Documentation', icon: BookOpen },
  ];

  return (
    <aside className="w-64 bg-slate-900/60 border-r border-slate-800 flex flex-col shrink-0">
      <div className="p-4 border-b border-slate-800/80">
        <div className="flex items-center space-x-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
          <Layers className="w-4 h-4 text-blue-400" />
          <span>Navigation Menu</span>
        </div>
      </div>

      <nav className="p-3 space-y-1.5 flex-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-slate-800 text-xs text-slate-500">
        <p className="font-semibold text-slate-400">Decoupled Architecture</p>
        <p className="mt-1">PyTorch &bull; FastAPI &bull; React &bull; Vite</p>
      </div>
    </aside>
  );
}
