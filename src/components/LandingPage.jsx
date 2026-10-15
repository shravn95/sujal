import React, { useEffect, useState } from 'react';
import { 
  BarChart3, Sparkles, UploadCloud, Database, Wand2, Search, MessageSquare, 
  LayoutDashboard, Lightbulb, TrendingUp, ChevronRight, Play, Sun, Moon 
} from 'lucide-react';
import { HeroDashboardShowcase } from './HeroDashboardShowcase';
import './LandingPage.css';

const AnimatedShowcaseChart = ({ type }) => {
  switch (type) {
    case 'line':
      return (
        <svg width="100%" height="100%" viewBox="0 0 240 120" style={{ overflow: 'visible' }}>
          <path d="M10 100 Q 60 20, 110 70 T 230 30" fill="none" stroke="var(--c-accent-blue)" strokeWidth="3" className="animated-path" />
          <path d="M10 100 Q 60 20, 110 70 T 230 30 L 230 110 L 10 110 Z" fill="url(#blueGrad)" opacity="0.25" />
          <circle cx="230" cy="30" r="5" fill="var(--c-accent-blue)" className="pulse-dot" />
          <defs>
            <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--c-accent-blue)" />
              <stop offset="100%" stopColor="transparent" />
            </linearGradient>
          </defs>
        </svg>
      );
    case 'donut':
      return (
        <svg width="100%" height="100%" viewBox="0 0 240 120">
          <circle cx="120" cy="60" r="40" fill="none" stroke="var(--c-glass-border)" strokeWidth="12" />
          <circle cx="120" cy="60" r="40" fill="none" stroke="var(--c-accent-purple)" strokeWidth="12" strokeDasharray="140 250" strokeDashoffset="0" className="rotating-donut" />
          <circle cx="120" cy="60" r="40" fill="none" stroke="var(--c-accent-cyan)" strokeWidth="12" strokeDasharray="60 250" strokeDashoffset="-140" />
        </svg>
      );
    case 'bars':
      return (
        <svg width="100%" height="100%" viewBox="0 0 240 120">
          {[
            { x: 30, h: 40, col: 'var(--c-accent-blue)' },
            { x: 70, h: 75, col: 'var(--c-accent-purple)' },
            { x: 110, h: 50, col: 'var(--c-accent-emerald)' },
            { x: 150, h: 90, col: 'var(--c-accent-cyan)' },
            { x: 190, h: 65, col: 'var(--c-accent-orange)' }
          ].map((bar, i) => (
            <rect key={i} x={bar.x} y={110 - bar.h} width="20" height={bar.h} rx="4" fill={bar.col} className={`growing-bar bar-${i}`} />
          ))}
        </svg>
      );
    case 'scatter':
      return (
        <svg width="100%" height="100%" viewBox="0 0 240 120">
          {[
            { cx: 40, cy: 80, r: 6, col: 'var(--c-accent-blue)' },
            { cx: 80, cy: 40, r: 10, col: 'var(--c-accent-purple)' },
            { cx: 130, cy: 70, r: 8, col: 'var(--c-accent-emerald)' },
            { cx: 170, cy: 30, r: 12, col: 'var(--c-accent-cyan)' },
            { cx: 210, cy: 90, r: 7, col: 'var(--c-accent-orange)' }
          ].map((dot, i) => (
            <circle key={i} cx={dot.cx} cy={dot.cy} r={dot.r} fill={dot.col} className={`floating-node node-${i}`} />
          ))}
        </svg>
      );
    case 'network':
      return (
        <svg width="100%" height="100%" viewBox="0 0 240 120">
          <line x1="40" y1="60" x2="120" y2="30" stroke="var(--c-glass-border)" strokeWidth="2" />
          <line x1="40" y1="60" x2="120" y2="90" stroke="var(--c-glass-border)" strokeWidth="2" />
          <line x1="120" y1="30" x2="200" y2="60" stroke="var(--c-glass-border)" strokeWidth="2" />
          <line x1="120" y1="90" x2="200" y2="60" stroke="var(--c-glass-border)" strokeWidth="2" />
          <circle cx="40" cy="60" r="10" fill="var(--c-accent-blue)" />
          <circle cx="120" cy="30" r="10" fill="var(--c-accent-purple)" />
          <circle cx="120" cy="90" r="10" fill="var(--c-accent-cyan)" />
          <circle cx="200" cy="60" r="12" fill="var(--c-accent-emerald)" className="pulse-dot" />
        </svg>
      );
    case 'forecast':
    default:
      return (
        <svg width="100%" height="100%" viewBox="0 0 240 120">
          <path d="M10 80 Q 70 90, 120 50" fill="none" stroke="var(--c-accent-emerald)" strokeWidth="3" />
          <path d="M120 50 Q 170 10, 230 40" fill="none" stroke="var(--c-accent-orange)" strokeWidth="3" strokeDasharray="4 4" className="animated-path" />
          <path d="M120 50 Q 170 -10, 230 20 L 230 70 Q 170 30, 120 50 Z" fill="var(--c-accent-orange)" opacity="0.15" />
        </svg>
      );
  }
};

const LandingPage = ({ onGetStarted, toggleTheme, theme }) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="landing-page">
      {/* Top Navigation */}
      <nav className={`top-nav ${scrolled ? 'nav-scrolled' : ''}`}>
        <div className="nav-container">
          <div className="nav-logo">
            <Sparkles className="logo-icon" size={24} />
            <span>AutoInsights</span>
          </div>
          <div className="nav-actions">
            <button className="theme-toggle" onClick={toggleTheme}>
              {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
            </button>
            <button className="btn btn-primary" onClick={onGetStarted}>Get Started</button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title animate-fade-in-up">Your AI Data Scientist.</h1>
          <p className="hero-subtitle animate-fade-in-up animate-delay-1">
            Upload any dataset. Ask questions naturally. Receive beautiful interactive dashboards instantly.
          </p>
          <div className="hero-buttons animate-fade-in-up animate-delay-2">
            <button className="btn btn-primary btn-large" onClick={onGetStarted}>
              Start Free <ChevronRight size={18} style={{marginLeft: '4px'}}/>
            </button>
            <button className="btn btn-glass btn-large">
              <Play size={18} style={{marginRight: '8px'}} fill="currentColor"/> Watch Demo
            </button>
          </div>
        </div>

        {/* Hero Illustration */}
        <HeroDashboardShowcase />
      </header>

      {/* Showcase Section */}
      <section className="showcase-section">
        <h2 className="section-title">Built for every team</h2>
        <div className="showcase-grid">
          {[
            { title: 'Sales Dashboard', type: 'line' },
            { title: 'Marketing Dashboard', type: 'donut' },
            { title: 'Finance Dashboard', type: 'bars' },
            { title: 'Customer Analytics', type: 'scatter' },
            { title: 'Supply Chain', type: 'network' },
            { title: 'Forecasting', type: 'forecast' }
          ].map((card, idx) => (
            <div key={idx} className="showcase-card glass-panel" onClick={onGetStarted}>
              <div className="card-mockup">
                <AnimatedShowcaseChart type={card.type} />
              </div>
              <h3>{card.title}</h3>
            </div>
          ))}
        </div>
      </section>

      {/* AI Workflow Section */}
      <section className="workflow-section">
        <h2 className="section-title">From Raw Data to Intelligence</h2>
        <div className="workflow-grid">
          {[
            { icon: <UploadCloud size={20} />, step: '01', title: 'Upload Dataset', desc: 'CSV, Excel, SQL, API, Cloud' },
            { icon: <Database size={20} />, step: '02', title: 'AI Understands Data', desc: 'Schema & type detection' },
            { icon: <Wand2 size={20} />, step: '03', title: 'Automatic Cleaning', desc: 'Missing values & outliers' },
            { icon: <Search size={20} />, step: '04', title: 'Automated EDA', desc: 'Distribution & correlation' },
            { icon: <MessageSquare size={20} />, step: '05', title: 'Natural Query', desc: 'Ask in plain language' },
            { icon: <LayoutDashboard size={20} />, step: '06', title: 'Instant Dashboard', desc: 'AI builds visual layout' },
            { icon: <Lightbulb size={20} />, step: '07', title: 'Interactive Insights', desc: 'Contextual tooltips' },
            { icon: <TrendingUp size={20} />, step: '08', title: 'Forecast & Recommend', desc: 'Predictive modeling' }
          ].map((item, i) => (
            <div key={i} className="workflow-compact-card glass-panel">
              <div className="compact-card-header">
                <div className="compact-icon-box">{item.icon}</div>
                <span className="step-pill">STEP {item.step}</span>
              </div>
              <h4>{item.title}</h4>
              <p>{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Smart Features Section */}
      <section className="features-section">
        <h2 className="section-title">Everything you need. Nothing you don't.</h2>
        <div className="features-grid">
          {[
            'AI Dashboard Generator', 'Natural Language Analytics', 'Automatic Data Cleaning', 
            'Automatic EDA', 'Forecasting', 'Anomaly Detection', 'Root Cause Analysis', 
            'Auto KPI Detection', 'Storytelling Reports', 'One-click Export', 
            'Interactive Sharing', 'Real-time Collaboration'
          ].map((feature, i) => (
            <div key={i} className="feature-item glass-panel">
              <Sparkles size={16} className="feature-icon" />
              <span>{feature}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer glass-panel">
        <div className="footer-content">
          <div className="footer-brand">
            <Sparkles size={20} />
            <span>AutoInsights</span>
          </div>
          <div className="footer-links">
            <a href="#">Privacy</a>
            <a href="#">Terms</a>
            <a href="#">Twitter</a>
            <a href="#">GitHub</a>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2030 AutoInsights Inc. Designed in California.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
