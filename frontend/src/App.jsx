import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import DiagnosticsPage from './pages/DiagnosticsPage';
import ModelStatusPage from './pages/ModelStatusPage';
import ResearchDocsPage from './pages/ResearchDocsPage';
import apiService from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [healthData, setHealthData] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchHealth = async () => {
    setIsRefreshing(true);
    const res = await apiService.getHealthStatus();
    if (res.success) {
      setHealthData(res.data);
    } else {
      setHealthData(null);
    }
    setIsRefreshing(false);
  };

  useEffect(() => {
    fetchHealth();
    // Periodically poll backend health every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100">
      {/* Top Navigation */}
      <Navbar systemHealth={healthData} />

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Content Viewport */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <div className="max-w-6xl mx-auto">
            {activeTab === 'dashboard' && (
              <Dashboard
                healthData={healthData}
                onRefresh={fetchHealth}
                setActiveTab={setActiveTab}
              />
            )}
            {activeTab === 'diagnostics' && (
              <DiagnosticsPage systemHealth={healthData} />
            )}
            {activeTab === 'models' && (
              <ModelStatusPage healthData={healthData} />
            )}
            {activeTab === 'docs' && (
              <ResearchDocsPage />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
