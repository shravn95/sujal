import { useState } from 'react';
import LandingPage from './components/LandingPage';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('landing'); // 'landing' or 'dashboard'
  const [theme, setTheme] = useState('light');

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  return (
    <div className={`app-container ${theme}`}>
      <div className="bg-mesh"></div>
      
      {currentPage === 'landing' && (
        <LandingPage 
          onGetStarted={() => setCurrentPage('dashboard')} 
          toggleTheme={toggleTheme} 
          theme={theme} 
        />
      )}
      
      {currentPage === 'dashboard' && (
        <Dashboard 
          onBack={() => setCurrentPage('landing')} 
          toggleTheme={toggleTheme} 
          theme={theme} 
        />
      )}
    </div>
  );
}

export default App;
