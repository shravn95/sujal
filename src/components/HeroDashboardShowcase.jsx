import React, { useState, useEffect } from 'react';
import { 
  Sparkles, CheckCircle2, FolderUp, Cpu, Wand2, BarChart2, Lightbulb, 
  TrendingUp, AlertCircle, ArrowUpRight, Check, Play, FileText, Database, 
  Search, ShieldCheck
} from 'lucide-react';
import './HeroDashboardShowcase.css';

export const HeroDashboardShowcase = () => {
  const [animStep, setAnimStep] = useState(0);
  const [typedPrompt, setTypedPrompt] = useState('');

  const fullPrompt = "Show yearly sales performance by region...";

  // 12-Second Master Animation Loop
  useEffect(() => {
    const timer = setInterval(() => {
      setAnimStep((prev) => (prev + 1) % 6);
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  // Handle typing animation when animStep === 3
  useEffect(() => {
    if (animStep === 3) {
      let index = 0;
      setTypedPrompt('');
      const typing = setInterval(() => {
        if (index < fullPrompt.length) {
          setTypedPrompt(fullPrompt.slice(0, index + 1));
          index++;
        } else {
          clearInterval(typing);
        }
      }, 40);
      return () => clearInterval(typing);
    } else if (animStep < 3) {
      setTypedPrompt('');
    }
  }, [animStep]);

  const workflowSteps = [
    { icon: <FolderUp size={16} />, title: "Dataset Uploaded", stepIndex: 0 },
    { icon: <Cpu size={16} />, title: "Understanding Schema", stepIndex: 1 },
    { icon: <Wand2 size={16} />, title: "Cleaning Data", stepIndex: 2 },
    { icon: <BarChart2 size={16} />, title: "Performing EDA", stepIndex: 3 },
    { icon: <Sparkles size={16} />, title: "Generating Dashboard", stepIndex: 4 },
    { icon: <TrendingUp size={16} />, title: "Forecast Ready", stepIndex: 5 },
  ];

  return (
    <div className="hero-showcase-container">
      {/* Outer macOS Frame */}
      <div className="showcase-mac-window glass-panel liquid-glass">
        
        {/* Window Top Bar & Floating AI Search Input */}
        <div className="mac-window-header">
          <div className="mac-dots">
            <span className="dot red"></span>
            <span className="dot yellow"></span>
            <span className="dot green"></span>
          </div>
          <div className="hero-search-bar glass-panel">
            <Sparkles size={16} className="search-sparkle-icon" />
            <span className="search-text">
              {animStep >= 3 ? typedPrompt : "Ask anything about your data..."}
            </span>
            {animStep >= 3 && <span className="typing-cursor">|</span>}
          </div>
          <div className="header-status-badge">
            <span className="status-dot-live"></span> AI Autonomous
          </div>
        </div>

        {/* Main Interface Layout */}
        <div className="showcase-grid-body">
          
          {/* LEFT SIDEBAR: AI Automation Workflow */}
          <aside className="hero-sidebar-workflow glass-panel">
            <div className="workflow-header">
              <Sparkles size={14} color="var(--c-accent-blue)" />
              <span>AI AGENT WORKFLOW</span>
            </div>
            <div className="workflow-steps-list">
              {workflowSteps.map((step) => {
                const isDone = animStep > step.stepIndex;
                const isActive = animStep === step.stepIndex;
                return (
                  <div 
                    key={step.stepIndex} 
                    className={`workflow-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}
                  >
                    <div className="step-status-icon">
                      {isDone ? (
                        <CheckCircle2 size={16} className="text-emerald" />
                      ) : (
                        step.icon
                      )}
                    </div>
                    <span className="step-title">{step.title}</span>
                    {isActive && <div className="active-glow-pulse" />}
                  </div>
                );
              })}
            </div>
          </aside>

          {/* CENTER MAIN DASHBOARD WORKSPACE */}
          <main className="hero-main-dashboard">
            
            {/* Top KPI Cards Row */}
            <div className="hero-kpi-row">
              <div className="hero-mini-kpi glass-panel">
                <span className="kpi-title">Revenue</span>
                <span className="kpi-val">₹4.8M</span>
                <span className="kpi-change positive">↑ 24%</span>
              </div>
              <div className="hero-mini-kpi glass-panel">
                <span className="kpi-title">Orders</span>
                <span className="kpi-val">18,452</span>
                <span className="kpi-change positive">↑ 12%</span>
              </div>
              <div className="hero-mini-kpi glass-panel">
                <span className="kpi-title">Profit</span>
                <span className="kpi-val">₹1.1M</span>
                <span className="kpi-change positive">↑ 16%</span>
              </div>
              <div className="hero-mini-kpi glass-panel">
                <span className="kpi-title">AOV</span>
                <span className="kpi-val">₹1,245</span>
                <span className="kpi-change positive">↑ 7%</span>
              </div>
            </div>

            {/* Central Visualizations Grid */}
            <div className="hero-charts-area">
              
              {/* Primary Dynamic Chart Card */}
              <div className="hero-chart-card glass-panel relative">
                <div className="chart-card-top">
                  <h4>{animStep >= 3 ? "Regional Revenue Comparison" : "Revenue Growth Trend"}</h4>
                  <span className="badge-live">Live Model</span>
                </div>

                <div className="hero-svg-wrapper">
                  {animStep < 3 ? (
                    // Default Revenue Area Chart
                    <svg width="100%" height="100%" viewBox="0 0 400 150">
                      <defs>
                        <linearGradient id="heroGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="var(--c-accent-blue)" stopOpacity="0.4" />
                          <stop offset="100%" stopColor="var(--c-accent-blue)" stopOpacity="0.0" />
                        </linearGradient>
                      </defs>
                      <line x1="30" y1="130" x2="380" y2="130" stroke="var(--c-glass-border)" strokeWidth="1" />
                      <line x1="30" y1="20" x2="30" y2="130" stroke="var(--c-glass-border)" strokeWidth="1" />
                      <path d="M30 110 Q 100 40, 180 80 T 330 30 L 380 45 L 380 130 L 30 130 Z" fill="url(#heroGrad)" />
                      <path d="M30 110 Q 100 40, 180 80 T 330 30 L 380 45" fill="none" stroke="var(--c-accent-blue)" strokeWidth="3" className="animated-path" />
                      <circle cx="330" cy="30" r="5" fill="var(--c-accent-blue)" className="pulse-dot" />
                    </svg>
                  ) : (
                    // Transformed Regional Bar Chart on Step >= 3
                    <svg width="100%" height="100%" viewBox="0 0 400 150">
                      <line x1="30" y1="130" x2="380" y2="130" stroke="var(--c-glass-border)" strokeWidth="1" />
                      {[
                        { reg: 'West', val: 110, col: 'var(--c-accent-blue)' },
                        { reg: 'North', val: 75, col: 'var(--c-accent-purple)' },
                        { reg: 'South', val: 90, col: 'var(--c-accent-emerald)' },
                        { reg: 'East', val: 50, col: 'var(--c-accent-cyan)' }
                      ].map((item, i) => (
                        <g key={i}>
                          <rect 
                            x={60 + i * 80} 
                            y={130 - item.val} 
                            width="36" 
                            height={item.val} 
                            rx="5" 
                            fill={item.col} 
                            className={`growing-bar bar-${i}`} 
                          />
                          <text x={78 + i * 80} y="145" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="10">{item.reg}</text>
                        </g>
                      ))}
                    </svg>
                  )}
                </div>

                {/* Floating macOS Sonoma Hover Tooltip on Step >= 4 */}
                {animStep >= 4 && (
                  <div className="hero-apple-tooltip glass-panel animate-fade-in-up">
                    <div className="tooltip-sparkle">
                      <Sparkles size={12} color="var(--c-accent-blue)" />
                      <span>AI INSIGHT VERIFIED</span>
                    </div>
                    <p className="tooltip-main-txt">West region generated <strong>34% of total revenue</strong>.</p>
                    <span className="tooltip-conf">Confidence Score: 97.4%</span>
                  </div>
                )}
              </div>

              {/* Secondary Bottom Grid: Donut + Insights Summary */}
              <div className="hero-sub-grid">
                
                {/* Customer Segments Donut */}
                <div className="hero-mini-card glass-panel">
                  <h5>Customer Segments</h5>
                  <div className="mini-donut-wrapper">
                    <svg width="100" height="100" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="34" fill="none" stroke="var(--c-glass-border)" strokeWidth="10" />
                      <circle cx="50" cy="50" r="34" fill="none" stroke="var(--c-accent-blue)" strokeWidth="10" strokeDasharray="100 220" />
                      <circle cx="50" cy="50" r="34" fill="none" stroke="var(--c-accent-purple)" strokeWidth="10" strokeDasharray="60 220" strokeDashoffset="-100" />
                      <circle cx="50" cy="50" r="34" fill="none" stroke="var(--c-accent-emerald)" strokeWidth="10" strokeDasharray="40 220" strokeDashoffset="-160" />
                    </svg>
                    <div className="donut-legend">
                      <div><span style={{background: 'var(--c-accent-blue)'}}></span> Enterprise (50%)</div>
                      <div><span style={{background: 'var(--c-accent-purple)'}}></span> SMB (30%)</div>
                      <div><span style={{background: 'var(--c-accent-emerald)'}}></span> Consumer (20%)</div>
                    </div>
                  </div>
                </div>

                {/* AI Executive Summary Card */}
                <div className="hero-mini-card glass-panel">
                  <h5><Lightbulb size={14} color="var(--c-accent-orange)" /> AI Opportunities</h5>
                  <div className="mini-insights-list">
                    <div className="insight-bullet">
                      <ArrowUpRight size={12} className="text-emerald" />
                      <span>Increase inventory in Maharashtra by 15%</span>
                    </div>
                    <div className="insight-bullet">
                      <AlertCircle size={12} className="text-orange" />
                      <span>April revenue dipped due to seasonal shift</span>
                    </div>
                  </div>
                </div>

              </div>

            </div>
          </main>

          {/* RIGHT SIDEBAR: AI Analyst Panel */}
          <aside className="hero-ai-panel glass-panel">
            <div className="ai-panel-header">
              <div className="ai-analyst-avatar">
                <Sparkles size={16} />
              </div>
              <div className="ai-analyst-title">
                <h4>AI Analyst</h4>
                <span className="thinking-indicator">
                  <span className="dot-pulse"></span> Understanding dataset...
                </span>
              </div>
            </div>

            {/* Dataset Metadata Box */}
            <div className="dataset-meta-box glass-panel">
              <div className="meta-filename">
                <FileText size={14} color="var(--c-accent-blue)" />
                <span>Retail_Sales_2030.csv</span>
              </div>
              <div className="meta-stats-row">
                <div className="meta-stat">
                  <span className="stat-num">2.4M</span>
                  <span className="stat-lbl">Rows</span>
                </div>
                <div className="meta-stat">
                  <span className="stat-num">42</span>
                  <span className="stat-lbl">Columns</span>
                </div>
              </div>
            </div>

            {/* Detected KPIs Chips */}
            <div className="detected-kpis-section">
              <span className="section-label">DETECTED KPIS</span>
              <div className="kpi-chips-grid">
                <span className="kpi-chip">Revenue</span>
                <span className="kpi-chip">Profit</span>
                <span className="kpi-chip">Orders</span>
                <span className="kpi-chip">CLV</span>
              </div>
            </div>

            {/* AI Suggestions Bullets */}
            <div className="ai-suggestions-section">
              <span className="section-label">AI SUGGESTIONS</span>
              <ul className="suggestions-list">
                <li><Sparkles size={12} /> Analyze yearly revenue trend</li>
                <li><Sparkles size={12} /> Customer segmentation model</li>
                <li><Sparkles size={12} /> Forecast next quarter sales</li>
                <li><Sparkles size={12} /> Detect regional anomalies</li>
              </ul>
            </div>

            {/* Action CTA Button */}
            <button className="btn btn-primary generate-cta-btn">
              <Sparkles size={16} /> Generate Executive Dashboard
            </button>
          </aside>

        </div>
      </div>

      {/* FLOATING NOTIFICATION BADGES (Apple VisionOS Style) */}
      <div className="hero-floating-badge badge-top-left glass-panel liquid-glass animate-float-delay">
        <TrendingUp size={18} color="var(--c-accent-emerald)" />
        <div className="badge-txt">
          <strong>Revenue +24%</strong>
          <span>Compared to last year • <ShieldCheck size={10} inline /> AI Verified</span>
        </div>
      </div>

      <div className="hero-floating-badge badge-bottom-right glass-panel liquid-glass animate-float">
        <Sparkles size={18} color="var(--c-accent-purple)" />
        <div className="badge-txt">
          <strong>AI Analysis Complete</strong>
          <span>12 Insights Found • Forecast Ready</span>
        </div>
      </div>

      {animStep >= 3 && (
        <div className="hero-floating-badge badge-bottom-left glass-panel liquid-glass animate-fade-in-up">
          <Search size={16} color="var(--c-accent-cyan)" />
          <div className="badge-txt">
            <strong>Prompt Triggered</strong>
            <span>"Show yearly sales by region..."</span>
          </div>
        </div>
      )}
    </div>
  );
};
