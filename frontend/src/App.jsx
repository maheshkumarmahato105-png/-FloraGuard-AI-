import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { ToastProvider } from './context/ToastContext';
import { useBackendHealth } from './hooks/useBackendHealth';
import Sidebar from './components/layout/Sidebar';
import Navbar from './components/layout/Navbar';
import MobileNavigation from './components/layout/MobileNavigation';
import ToastContainer from './components/common/ToastContainer';
import FloraChatDrawer from './components/common/FloraChatDrawer';
import Dashboard from './pages/Dashboard';
import Detection from './pages/Detection';
import Results from './pages/Results';
import History from './pages/History';
import Settings from './pages/Settings';
import SampleReport from './pages/SampleReport';
import NotFound from './pages/NotFound';

export function AppShell() {
  const healthState = useBackendHealth({ autoFetch: true });

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      <Sidebar
        health={healthState.health}
        systemInfo={healthState.systemInfo}
        loading={healthState.loading}
      />

      <div className="app-main-area">
        <Navbar
          health={healthState.health}
          loading={healthState.loading}
          onRefreshHealth={healthState.refresh}
        />

        <Routes>
          <Route path="/" element={<Dashboard healthState={healthState} />} />
          <Route path="/detection" element={<Detection healthState={healthState} />} />
          <Route path="/results/:requestId" element={<Results />} />
          <Route path="/history" element={<History />} />
          <Route path="/settings" element={<Settings healthState={healthState} />} />
          <Route path="/sample-report" element={<SampleReport />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>

      <MobileNavigation />
      <FloraChatDrawer />
      <ToastContainer />
    </div>
  );
}

export function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <BrowserRouter>
          <AppShell />
        </BrowserRouter>
      </ToastProvider>
    </ThemeProvider>
  );
}

export default App;
