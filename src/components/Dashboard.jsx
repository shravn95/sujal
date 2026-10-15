import React, { useState, useRef, useEffect } from "react";
import {
  LayoutDashboard,
  Database,
  MessageSquare,
  LineChart,
  FileText,
  Settings,
  ChevronLeft,
  Sparkles,
  Maximize2,
  X,
  AlertCircle,
  TrendingUp,
  ChevronDown,
  Upload,
  Plus,
  Trash2,
  Search,
  Download,
  Copy,
  Check,
  BarChart2,
  Clock,
  Activity,
  Cpu,
  Zap,
  Loader2,
} from "lucide-react";
import {
  ChartRenderer,
  CATEGORIZED_CHARTS,
  AVAILABLE_CHARTS,
} from "./ChartComponents";
import {
  uploadDataset,
  retryPipeline,
  STEPS, // ← add these two
  streamAIQuery,
  fetchDatasetPreview,
  preprocessDataset,
  downloadDataset,
  fetchHealth,
  exportReportPDF,
  selectPrimaryKpi,
  aggregateChart,
  resumePipeline,
  listDatasets,
  openDataset,
  deleteDataset,
  saveMessage,
  saveDashboardState,
} from "../api/client";
import "./Dashboard.css";

// Chart Chooser Dropdown Component for single cards
const ChartChooser = ({ currentType, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="chart-chooser" ref={dropdownRef}>
      <button
        className="chooser-btn glass-panel"
        onClick={() => setIsOpen(!isOpen)}
      >
        {currentType} <ChevronDown size={14} />
      </button>
      {isOpen && (
        <div
          className="chooser-menu glass-panel animate-fade-in-up"
          style={{
            animationDuration: "0.2s",
            maxHeight: "320px",
            width: "220px",
          }}
        >
          {Object.entries(CATEGORIZED_CHARTS).map(([category, charts]) => (
            <div key={category} className="chooser-category">
              <div className="chooser-category-title">{category}</div>
              {charts.map((type) => (
                <button
                  key={type}
                  className={`chooser-item ${currentType === type ? "active" : ""}`}
                  onClick={() => {
                    onSelect(type);
                    setIsOpen(false);
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Dynamic Chart Card Component
// Dynamic Chart Card Component (2-Row Header + Guaranteed Non-Empty Axis Dropdowns)
const ChartCard = ({
  chart,
  isExpanded,
  onToggleExpand,
  onRemove,
  onTypeChange,
  chartConfig,
  onAxisChange,
  datasetSummary,
  isThinking,
}) => {
  const config = chartConfig || {};
  const currentX = config.x_axis || chart.x_axis || chart.x_column || "";
  const currentY = config.y_axis || chart.y_axis || chart.y_column || "";
  const currentAgg = (
    config.aggregation ||
    chart.aggregation ||
    "sum"
  ).toLowerCase();

  // Build dimension options and ALWAYS include currentX so dropdown is never blank
  const getValidDimensions = () => {
    const cats = datasetSummary?.categorical_columns || [];
    const dates =
      datasetSummary?.date_columns || datasetSummary?.datetime_columns || [];
    const numeric = datasetSummary?.numeric_columns || [];
    const ids = datasetSummary?.identifier_columns || [];

    const type = (chart.type || "").toLowerCase();
    let pool = [];

    if (type.includes("scatter")) {
      pool = [...numeric];
    } else if (type.includes("histogram") || type.includes("box")) {
      pool = ["Auto-binned", ...cats];
    } else {
      // Combine dates, categoricals, and low-cardinality numerics (like 'year')
      pool = [...dates, ...cats, ...numeric];
    }

    const filtered = Array.from(new Set(pool.filter((c) => !ids.includes(c))));
    if (currentX && !filtered.includes(currentX)) {
      filtered.unshift(currentX);
    }
    return filtered.length > 0 ? filtered : [currentX || "Dimension"];
  };

  // Build metric options and ALWAYS include currentY so dropdown is never blank
  const getValidMetrics = () => {
    const numeric = datasetSummary?.numeric_columns || [];
    const ids = datasetSummary?.identifier_columns || [];
    const filtered = Array.from(
      new Set(numeric.filter((n) => !ids.includes(n))),
    );
    if (currentY && !filtered.includes(currentY)) {
      filtered.unshift(currentY);
    }
    return filtered.length > 0 ? filtered : [currentY || "Value"];
  };

  const validDimensions = getValidDimensions();
  const validMetrics = getValidMetrics();

  return (
    <div className={`chart-card glass-panel ${isExpanded ? "expanded" : ""}`}>
      {/* Row 1: Full Chart Title + Expand/Remove Buttons */}
      <div className="card-header-top">
        <h3 title={chart.title}>{chart.title}</h3>
        <div className="card-actions">
          <button
            className="icon-btn small"
            onClick={() => onToggleExpand(chart.id)}
            title={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <X size={15} /> : <Maximize2 size={15} />}
          </button>
          <button
            className="icon-btn small"
            onClick={() => onRemove(chart.id)}
            title="Remove Chart"
          >
            <Trash2 size={15} style={{ color: "var(--c-accent-orange)" }} />
          </button>
        </div>
      </div>

      {/* Row 2: Chart Type + X-Axis -> Y-Axis + Aggregation */}
      <div className="card-controls-bar">
        <ChartChooser
          currentType={chart.type}
          onSelect={(newType) => onTypeChange(chart.id, newType)}
        />

        <select
          className="control-select"
          value={currentX}
          onChange={(e) => onAxisChange(chart.id, { x_axis: e.target.value })}
          disabled={isThinking}
          title={`X-Axis: ${currentX}`}
        >
          {validDimensions.map((dim) => (
            <option key={dim} value={dim}>
              {dim}
            </option>
          ))}
        </select>

        <span className="axis-arrow">→</span>

        <select
          className="control-select"
          value={currentY}
          onChange={(e) => onAxisChange(chart.id, { y_axis: e.target.value })}
          disabled={isThinking}
          title={`Y-Axis: ${currentY}`}
        >
          {validMetrics.map((metric) => (
            <option key={metric} value={metric}>
              {metric}
            </option>
          ))}
        </select>

        <select
          className="control-select agg-badge"
          value={currentAgg === "avg" ? "mean" : currentAgg}
          onChange={(e) =>
            onAxisChange(chart.id, { aggregation: e.target.value })
          }
          disabled={isThinking}
          title="Aggregation Function"
        >
          <option value="sum">SUM</option>
          <option value="mean">AVG</option>
          <option value="median">MEDIAN</option>
          <option value="count">COUNT</option>
          <option value="min">MIN</option>
          <option value="max">MAX</option>
        </select>
      </div>

      {/* Main Chart Plot */}
      <div className="chart-container">
        <ChartRenderer
          type={chart.type}
          data={chart.data}
          xAxis={currentX}
          yAxis={currentY}
          secondaryDimension={chart.secondary_dimension}
          matrixData={chart.matrix_data}
        />
      </div>

      {/* Insight Footer */}
      {(chart.insight_tooltip || chart.description) && (
        <div className="tooltip-insight">
          <Sparkles
            size={14}
            color="var(--c-accent-orange)"
            style={{ marginTop: "2px", flexShrink: 0 }}
          />
          <span>{chart.insight_tooltip || chart.description}</span>
        </div>
      )}

      {isExpanded && (
        <div className="expanded-insights animate-fade-in-up">
          <h4>Analytical Deep Dive ({chart.type})</h4>
          <p style={{ fontSize: "0.86rem", color: "var(--c-text-secondary)" }}>
            Analyzing <strong>{currentY || "Metric"}</strong> (
            {currentAgg.toUpperCase()}) segmented across{" "}
            <strong>{currentX || "Dimension"}</strong>.
          </p>
          <div className="insight-tags">
            <span className="tag success">
              <TrendingUp size={12} /> Verified Computation
            </span>
            <span className="tag warning">
              <AlertCircle size={12} /> Cleaned & Validated
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// Add Chart Modal Component
const AddChartModal = ({ isOpen, onClose, onAddChart }) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  if (!isOpen) return null;

  const categories = ["All", ...Object.keys(CATEGORIZED_CHARTS)];

  const getFilteredCharts = () => {
    let list = [];
    if (activeCategory === "All") {
      list = AVAILABLE_CHARTS;
    } else {
      list = CATEGORIZED_CHARTS[activeCategory] || [];
    }

    if (searchTerm.trim()) {
      list = list.filter((c) =>
        c.toLowerCase().includes(searchTerm.toLowerCase()),
      );
    }
    return list;
  };

  return (
    <div
      className="modal-backdrop glass-panel"
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 1000,
        background: "rgba(0,0,0,0.6)",
        backdropFilter: "blur(8px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
      }}
    >
      <div
        className="glass-panel"
        style={{
          width: "100%",
          maxWidth: "640px",
          maxHeight: "80vh",
          display: "flex",
          flexDirection: "column",
          borderRadius: "24px",
          padding: "24px",
          background: "var(--c-bg)",
          border: "1px solid var(--c-glass-border)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "16px",
          }}
        >
          <h3 style={{ fontSize: "1.25rem", fontWeight: 600 }}>
            Select Visualization Type
          </h3>
          <button className="icon-btn small" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Search & Category Filter */}
        <div style={{ display: "flex", gap: "12px", marginBottom: "16px" }}>
          <div style={{ position: "relative", flex: 1 }}>
            <Search
              size={16}
              style={{
                position: "absolute",
                left: "12px",
                top: "12px",
                color: "var(--c-text-secondary)",
              }}
            />
            <input
              type="text"
              placeholder="Search visualization catalog..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: "100%",
                padding: "10px 12px 10px 36px",
                borderRadius: "12px",
                border: "1px solid var(--c-glass-border)",
                background: "var(--c-glass-bg)",
                color: "var(--c-text)",
                outline: "none",
              }}
            />
          </div>
        </div>

        <div
          style={{
            display: "flex",
            gap: "8px",
            overflowX: "auto",
            paddingBottom: "12px",
            marginBottom: "12px",
          }}
        >
          {categories.map((cat) => (
            <button
              key={cat}
              className={`btn ${activeCategory === cat ? "btn-primary" : "btn-glass"}`}
              style={{
                padding: "6px 14px",
                fontSize: "0.82rem",
                whiteSpace: "nowrap",
              }}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Chart List Grid */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))",
            gap: "10px",
            paddingRight: "4px",
          }}
        >
          {getFilteredCharts().map((type) => (
            <button
              key={type}
              className="glass-panel"
              style={{
                padding: "12px",
                borderRadius: "12px",
                border: "1px solid var(--c-glass-border)",
                textAlign: "left",
                background: "var(--c-glass-bg)",
                cursor: "pointer",
                transition: "all 0.2s",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
              }}
              onClick={() => {
                onAddChart(type);
                onClose();
              }}
            >
              <span
                style={{
                  fontWeight: 600,
                  fontSize: "0.9rem",
                  color: "var(--c-text)",
                }}
              >
                {type}
              </span>
              <span
                style={{
                  fontSize: "0.75rem",
                  color: "var(--c-text-secondary)",
                }}
              >
                Click to materialize
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// Executive Report Modal Component with Markdown & Export
const ReportModal = ({ isOpen, onClose, report, datasetName }) => {
  const [copied, setCopied] = useState(false);
  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(report || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([report || ""], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${datasetName || "executive"}_insights_report.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className="modal-backdrop glass-panel"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        background: "rgba(0,0,0,0.75)",
        backdropFilter: "blur(12px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
        boxSizing: "border-box",
      }}
    >
      <section
        className="glass-panel animate-fade-in-up"
        style={{
          width: "min(860px, 95vw)",
          maxHeight: "85vh",
          borderRadius: "20px",
          padding: "24px 28px",
          background: "var(--c-bg)",
          border: "1px solid var(--c-glass-border)",
          display: "flex",
          flexDirection: "column",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
          boxSizing: "border-box",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "16px",
            borderBottom: "1px solid var(--c-glass-border)",
            paddingBottom: "14px",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <FileText size={20} color="var(--c-accent-blue)" />
            <h3 style={{ fontSize: "1.2rem", fontWeight: 700 }}>
              Executive Analysis Report
            </h3>
            {datasetName && (
              <span
                style={{
                  fontSize: "0.75rem",
                  padding: "2px 8px",
                  borderRadius: "6px",
                  background: "rgba(139,92,246,0.12)",
                  color: "var(--c-accent-purple)",
                  fontWeight: 500,
                }}
              >
                {datasetName}
              </span>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              className="btn btn-glass"
              onClick={handleCopy}
              style={{
                padding: "6px 12px",
                fontSize: "0.82rem",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              {copied ? (
                <Check size={14} color="var(--c-accent-emerald)" />
              ) : (
                <Copy size={14} />
              )}
              {copied ? "Copied" : "Copy"}
            </button>
            <button
              className="btn btn-glass"
              onClick={handleDownload}
              style={{
                padding: "6px 12px",
                fontSize: "0.82rem",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <Download size={14} /> Download MD
            </button>
            <button className="icon-btn small" onClick={onClose}>
              <X size={18} />
            </button>
          </div>
        </div>

        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflowY: "auto",
            whiteSpace: "pre-wrap",
            lineHeight: 1.7,
            color: "var(--c-text)",
            fontSize: "0.92rem",
            paddingRight: "12px",
            wordBreak: "break-word",
            overflowWrap: "break-word",
            fontFamily: "system-ui, -apple-system, sans-serif",
          }}
        >
          {report ||
            "Upload a dataset and ask analytical questions to generate an executive report."}
        </div>
      </section>
    </div>
  );
};

const ACTIVE_KEY = "autoinsight:activeDataset";
const WELCOME_MESSAGE = {
  role: "ai",
  text: "Welcome to AutoInsight! I am your AI Data Scientist powered by NVIDIA Nemotron-3 Ultra 550B. Upload any CSV/Excel file or ask any question to generate interactive dashboards, automated EDA, and executive insights.",
};
const STATE_LABEL = {
  ready: "Ready",
  analyzing: "Analysing…",
  analyzed: "Building…",
  uploaded: "Uploaded",
  failed: "Failed",
};

const PipelineProgress = ({ steps, error, canRetry, onRetry }) => (
  <div className="live-reasoning-card animate-fade-in-up">
    <div className="reasoning-header">
      <div className="reasoning-title-group">
        {!error && <div className="ai-pulse-dot" />}
        <span>{error ? "Processing stopped" : "Processing dataset"}</span>
      </div>
    </div>

    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        padding: "4px 0",
      }}
    >
      {STEPS.map((s) => {
        const st = steps[s.id];
        const icon =
          st?.status === "done" ? (
            <Check size={14} color="var(--c-accent-emerald)" />
          ) : st?.status === "running" ? (
            <Loader2
              size={14}
              className="animate-spin"
              color="var(--c-accent-purple)"
            />
          ) : st?.status === "error" ? (
            <AlertCircle size={14} color="var(--c-accent-orange)" />
          ) : (
            <span style={{ width: 14, textAlign: "center", opacity: 0.4 }}>
              ○
            </span>
          );
        return (
          <div
            key={s.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              fontSize: "0.82rem",
              opacity: st ? 1 : 0.5,
            }}
          >
            {icon}
            <div style={{ flex: 1 }}>
              <div>{s.label}</div>
              {st?.status === "running" && st.detail && (
                <div
                  style={{
                    fontSize: "0.7rem",
                    color: "var(--c-text-secondary)",
                  }}
                >
                  {st.detail}
                </div>
              )}
            </div>
            {st?.seconds != null && (
              <span
                style={{
                  fontSize: "0.72rem",
                  color: "var(--c-text-secondary)",
                }}
              >
                {st.seconds}s
              </span>
            )}
          </div>
        );
      })}
    </div>

    {error && (
      <div style={{ marginTop: "8px", fontSize: "0.8rem" }}>
        <div style={{ color: "var(--c-accent-orange)" }}>{error.message}</div>
        {canRetry ? (
          <button
            className="btn btn-glass"
            style={{
              marginTop: "8px",
              padding: "4px 12px",
              fontSize: "0.78rem",
            }}
            onClick={onRetry}
          >
            Retry this step
          </button>
        ) : (
          <div style={{ marginTop: "6px", color: "var(--c-text-secondary)" }}>
            Select the file again to retry.
          </div>
        )}
      </div>
    )}
  </div>
);

const DatasetPicker = ({ datasets, activeId, onOpen, onDelete, disabled }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const close = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const active = datasets.find((d) => d.dataset_id === activeId);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        className="btn btn-glass"
        disabled={disabled || datasets.length === 0}
        onClick={() => setOpen((o) => !o)}
        style={{ gap: "6px" }}
      >
        <Database size={16} /> {active ? active.title : "My datasets"}{" "}
        <ChevronDown size={14} />
      </button>
      {open && (
        <div
          className="glass-panel animate-fade-in-up"
          style={{
            position: "absolute",
            right: 0,
            top: "110%",
            zIndex: 50,
            width: "320px",
            maxHeight: "360px",
            overflowY: "auto",
            padding: "6px",
            borderRadius: "14px",
            background: "var(--c-bg)",
            border: "1px solid var(--c-glass-border)",
          }}
        >
          {datasets.map((d) => (
            <div
              key={d.dataset_id}
              onClick={() => {
                setOpen(false);
                onOpen(d.dataset_id);
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 10px",
                borderRadius: "10px",
                cursor: "pointer",
                background:
                  d.dataset_id === activeId
                    ? "var(--c-glass-bg)"
                    : "transparent",
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                <div
                  style={{
                    fontSize: "0.85rem",
                    fontWeight: 600,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {d.title}
                </div>
                <div
                  style={{
                    fontSize: "0.7rem",
                    color: "var(--c-text-secondary)",
                  }}
                >
                  {(d.row_count ?? 0).toLocaleString()} rows ·{" "}
                  {new Date(d.created_at).toLocaleDateString()} ·{" "}
                  {STATE_LABEL[d.state] || d.state}
                </div>
              </div>
              <button
                className="icon-btn small"
                title="Delete dataset"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(d);
                }}
              >
                <Trash2 size={14} style={{ color: "var(--c-accent-orange)" }} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Zero-dependency Markdown & Table Renderer for Copilot Chat & Reports
const formatInlineMarkdown = (text) => {
  if (!text) return null;
  // Split by **bold** and `code`
  const parts = String(text).split(/(\*\*.*?\*\*|`.*?`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="md-bold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={i} className="md-inline-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
};

const MarkdownRenderer = ({ content }) => {
  if (!content) return null;
  const lines = String(content).split("\n");
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i].trim();

    // Skip empty lines
    if (!line) {
      i++;
      continue;
    }

    // Horizontal Rule (---)
    if (/^-{3,}$/.test(line)) {
      elements.push(<hr key={`hr-${i}`} className="md-hr" />);
      i++;
      continue;
    }

    // Code Block (```sql ... ```)
    if (line.startsWith("```")) {
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <pre key={`code-${i}`} className="md-code-block">
          <code>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    // Markdown Table (| Col1 | Col2 |)
    if (line.startsWith("|") && line.endsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i].trim());
        i++;
      }
      if (tableLines.length >= 2) {
        const parseRow = (r) =>
          r
            .split("|")
            .slice(1, -1)
            .map((cell) => cell.trim());
        const headers = parseRow(tableLines[0]);
        // Skip separator row (| :--- | :--- |)
        const dataRows = tableLines
          .slice(1)
          .filter((r) => !/^\|[\s:-]+\|$/.test(r.replace(/\|/g, "|")))
          .filter((r) => !r.includes("---"))
          .map(parseRow);

        elements.push(
          <div key={`tbl-${i}`} className="md-table-wrapper">
            <table className="md-table">
              <thead>
                <tr>
                  {headers.map((h, hi) => (
                    <th key={hi}>{formatInlineMarkdown(h)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {dataRows.map((row, ri) => (
                  <tr key={ri}>
                    {row.map((cell, ci) => (
                      <td key={ci}>{formatInlineMarkdown(cell)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>,
        );
        continue;
      }
    }

    // Headings (###, ####, etc.)
    const headingMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      elements.push(
        <div key={`h-${i}`} className={`md-heading md-h${level}`}>
          {formatInlineMarkdown(text)}
        </div>,
      );
      i++;
      continue;
    }

    // Bullet Points (-, *, •)
    if (/^([-*•]|\d+\.)\s+/.test(line)) {
      const listItems = [];
      while (i < lines.length && /^([-*•]|\d+\.)\s+/.test(lines[i].trim())) {
        listItems.push(lines[i].trim().replace(/^([-*•]|\d+\.)\s+/, ""));
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} className="md-list">
          {listItems.map((item, li) => (
            <li key={li}>{formatInlineMarkdown(item)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    // Standard Paragraph
    elements.push(
      <p key={`p-${i}`} className="md-paragraph">
        {formatInlineMarkdown(line)}
      </p>,
    );
    i++;
  }

  return <div className="md-content">{elements}</div>;
};

// Main Dashboard Component
const Dashboard = ({ onBack, toggleTheme, theme }) => {
  const [expandedCard, setExpandedCard] = useState(null);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState([
    {
      role: "ai",
      text: "Welcome to AutoInsight! I am your AI Data Scientist powered by NVIDIA Nemotron-3 Ultra 550B. Upload any CSV/Excel file or ask any question to generate interactive dashboards, automated EDA, and executive insights.",
    },
  ]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [activeReasoningText, setActiveReasoningText] = useState("");

  const [dashboardTitle, setDashboardTitle] = useState(
    "Upload a dataset to begin",
  );
  const [activeCharts, setActiveCharts] = useState([]);

  const [kpiSummary, setKpiSummary] = useState({
    primary_kpi: "—",
    value: "—",
    data_quality: "—",
    total_rows: "—",
  });

  // New state for KPI selection and chart configuration
  const [availableKpiOptions, setAvailableKpiOptions] = useState([]);
  const [selectedKpiColumn, setSelectedKpiColumn] = useState(null);
  const [chartConfigs, setChartConfigs] = useState({}); // Per-chart axis/aggregation config

  const [datasetSummary, setDatasetSummary] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  const [detailedReport, setDetailedReport] = useState("");
  const [forecast, setForecast] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [activeView, setActiveView] = useState("dashboard");
  const [previewData, setPreviewData] = useState({
    columns: [],
    rows: [],
    total_rows: 0,
    page: 1,
    page_size: 50,
    total_pages: 0,
    cleaning_summary: null,
  });
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [currentFileName, setCurrentFileName] = useState("");
  const [isDownloading, setIsDownloading] = useState(false);
  const [planSource, setPlanSource] = useState(null);
  const [cleaningExpanded, setCleaningExpanded] = useState(false);
  const [pipelineSteps, setPipelineSteps] = useState({});
  const [pipelineError, setPipelineError] = useState(null);
  const [pendingDatasetId, setPendingDatasetId] = useState(null);
  const stepStartRef = useRef({});
  const uploadFileNameRef = useRef("");
  const [datasets, setDatasets] = useState([]);
  const [isRestoring, setIsRestoring] = useState(true);
  const activeIdRef = useRef(null); // always the current dataset id, usable inside async callbacks
  const bootedRef = useRef(false);

  const runningStep = STEPS.find(
    (s) => pipelineSteps[s.id]?.status === "running",
  );

  const chatEndRef = useRef(null);

  // Restore the last dataset on page load
  useEffect(() => {
    if (bootedRef.current) return; // React StrictMode runs effects twice in dev
    bootedRef.current = true;
    (async () => {
      try {
        const { datasets: list } = await listDatasets();
        setDatasets(list);
        const last = localStorage.getItem(ACTIVE_KEY);
        const target = list.find((d) => d.dataset_id === last) || list[0];
        if (target) {
          await openWorkspace(target.dataset_id);
          return;
        }
      } catch (err) {
        console.warn("Could not restore workspace:", err.message);
      }
      setIsRestoring(false);
    })();
  }, []);

  // Save chart edits (removed charts, changed types or axes) shortly after the last change
  useEffect(() => {
    if (!datasetId || activeCharts.length === 0) return;
    const t = setTimeout(() => {
      saveDashboardState(datasetId, {
        charts: activeCharts.map((c) => ({
          ...c,
          aggregation: chartConfigs[c.id]?.aggregation,
        })),
      }).catch(() => {});
    }, 1000);
    return () => clearTimeout(t);
  }, [activeCharts, chartConfigs, datasetId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, activeReasoningText, pipelineSteps, pipelineError]);

  useEffect(() => {
    let interval = null;
    if (isThinking || isUploading || isDownloading) {
      setElapsedTime(0);
      interval = setInterval(() => {
        setElapsedTime((prev) => +(prev + 0.1).toFixed(1));
      }, 100);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isThinking, isUploading, isDownloading]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, activeReasoningText, pipelineSteps, pipelineError]);

  const handleToggleExpand = (id) => {
    setExpandedCard(expandedCard === id ? null : id);
  };

  const handleRemoveChart = (id) => {
    setActiveCharts((prev) => prev.filter((c) => c.id !== id));
  };

  const handleTypeChange = (id, newType) => {
    setActiveCharts((prev) =>
      prev.map((c) => (c.id === id ? { ...c, type: newType } : c)),
    );
  };

  const handleAddChart = (chartType) => {
    const newId = `custom-${Date.now()}`;
    const baseChart = activeCharts[0];
    setActiveCharts((prev) => [
      ...prev,
      {
        id: newId,
        title: `${chartType} Analysis`,
        type: chartType,
        x_axis: baseChart?.x_axis,
        y_axis: baseChart?.y_axis,
        secondary_dimension: baseChart?.secondary_dimension,
        data: baseChart?.data || [],
        insight_tooltip: `Dynamic visual representation in a ${chartType.toLowerCase()} view.`,
      },
    ]);
  };

  const refreshDatasets = () =>
    listDatasets()
      .then((r) => setDatasets(r.datasets))
      .catch(() => {});

  // Show a message and persist it to the server
  const addMessage = (msg, idOverride) => {
    setChatHistory((prev) => [...prev, msg]);
    const id = idOverride || activeIdRef.current;
    if (id)
      saveMessage(id, msg.role, msg.text, {
        sourceLabel: msg.sourceLabel,
        type: msg.type,
      }).catch(() => {});
  };

  const resetWorkspaceUI = () => {
    setActiveCharts([]);
    setChartConfigs({});
    setDatasetSummary(null);
    setDatasetId(null);
    setKpiSummary({
      primary_kpi: "—",
      value: "—",
      data_quality: "—",
      total_rows: "—",
    });
    setAvailableKpiOptions([]);
    setSelectedKpiColumn(null);
    setDashboardTitle("Upload a dataset to begin");
    setCurrentFileName("");
    setDetailedReport("");
    setForecast(null);
    setPipelineSteps({});
    setPipelineError(null);
    setPendingDatasetId(null);
    setPreviewData({
      columns: [],
      rows: [],
      total_rows: 0,
      page: 1,
      page_size: 50,
      total_pages: 0,
      cleaning_summary: null,
    });
    setActiveView("dashboard");
    setChatHistory([WELCOME_MESSAGE]);
  };

  const openWorkspace = async (id) => {
    setIsRestoring(true);
    resetWorkspaceUI();
    try {
      const ws = await openDataset(id);
      activeIdRef.current = id;
      localStorage.setItem(ACTIVE_KEY, id);
      uploadFileNameRef.current = ws.filename;
      setChatHistory([
        WELCOME_MESSAGE,
        ...ws.messages.map((m) => ({ role: m.role, text: m.text, ...m.meta })),
      ]);

      if (ws.state === "ready" && ws.dashboard) {
        processDatasetResult(ws.dashboard, ws.filename, { restore: true });
        return;
      }

      // Unfinished pipeline: continue where it stopped instead of starting over
      setPendingDatasetId(id);
      setCurrentFileName(ws.filename);
      if (ws.state === "failed") {
        setPipelineError({
          id: "analyze",
          message: "The previous analysis failed.",
        });
        return;
      }
      setIsUploading(true);
      try {
        const res = await resumePipeline(id, ws.state, handleStep);
        processDatasetResult(res, ws.filename);
        setPipelineSteps({});
      } catch {
        /* shown in the progress card */
      } finally {
        setIsUploading(false);
      }
    } catch (err) {
      setChatHistory([
        WELCOME_MESSAGE,
        {
          role: "ai",
          type: "error",
          text: `Could not open dataset: ${err.message}`,
        },
      ]);
    } finally {
      setIsRestoring(false);
    }
  };

  const handleDeleteDataset = async (d) => {
    if (!window.confirm(`Delete "${d.title}" and its chat history?`)) return;
    try {
      await deleteDataset(d.dataset_id);
      if (d.dataset_id === activeIdRef.current) {
        activeIdRef.current = null;
        localStorage.removeItem(ACTIVE_KEY);
        resetWorkspaceUI();
      }
      refreshDatasets();
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        { role: "ai", type: "error", text: `Delete failed: ${err.message}` },
      ]);
    }
  };

  const handleStep = (id, status, payload) => {
    const now = performance.now();
    if (status === "retrying") {
      setPipelineSteps((prev) => ({
        ...prev,
        [id]: { status: "running", seconds: null, detail: payload },
      }));
      return;
    }
    if (status === "running") stepStartRef.current[id] = now;

    setPipelineSteps((prev) => ({
      ...prev,
      [id]: {
        status,
        seconds:
          status === "running"
            ? null
            : +((now - stepStartRef.current[id]) / 1000).toFixed(1),
      },
    }));

    if (status === "error") {
      setPipelineError({ id, message: payload });
      return;
    }
    if (status !== "done") return;

    // Show real results in chat as each step finishes
    if (id === "upload") {
      activeIdRef.current = payload.dataset_id;
      setPendingDatasetId(payload.dataset_id);
      const r = payload.cleaning_report;
      addMessage(
        {
          role: "ai",
          text:
            `🧹 Cleaned ${payload.row_count.toLocaleString()} rows × ${payload.column_count} columns. ` +
            `${r.duplicates_removed} duplicates removed, ${r.type_conversions.length} type conversions.`,
        },
        payload.dataset_id,
      );
      refreshDatasets(); // the dataset appears in the picker right away
    }
    if (id === "analyze") {
      setPlanSource(payload.plan_source);
      addMessage({
        role: "ai",
        text: `${payload.executive_summary}\n\n${payload.key_observations.map((o) => `• ${o}`).join("\n")}`,
        sourceLabel:
          payload.plan_source === "llm"
            ? "✨ Nemotron-3 Ultra 550B"
            : "⚙️ Rule-based fallback",
      });
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    e.target.value = ""; // lets you re-select the same file after a failure
    if (!file) return;

    uploadFileNameRef.current = file.name;
    setPipelineSteps({});
    setPipelineError(null);
    setPendingDatasetId(null);
    setIsUploading(true);
    try {
      const res = await uploadDataset(file, handleStep);
      processDatasetResult(res, file.name);
      setPipelineSteps({});
    } catch {
      /* the failed step is already shown in the progress card */
    } finally {
      setIsUploading(false);
    }
  };

  const handleRetry = async () => {
    if (!pipelineError || !pendingDatasetId) return;
    const from = pipelineError.id;
    setPipelineError(null);
    setIsUploading(true);
    try {
      const res = await retryPipeline(pendingDatasetId, from, handleStep);
      processDatasetResult(res, uploadFileNameRef.current);
      setPipelineSteps({});
    } catch {
      /* shown in the card */
    } finally {
      setIsUploading(false);
    }
  };

  // Dynamic suggested prompts derived from the actual dataset schema
  const getSuggestedPrompts = () => {
    if (!datasetSummary) {
      return [{ label: "Upload a dataset to get started", prompt: "" }];
    }
    const kpi = datasetSummary.primary_kpi || "primary metric";
    const dates = datasetSummary.date_columns || [];
    const cats = datasetSummary.categorical_columns || [];
    const nums = datasetSummary.numeric_columns || [];
    const prompts = [];

    prompts.push({
      label: `Give me a complete overview of the dataset`,
      prompt: "Give me a complete overview and analysis of the entire dataset",
    });

    if (dates.length > 0) {
      prompts.push({
        label: `Show ${kpi} trend over time`,
        prompt: `Show the trend of ${kpi} over ${dates[0]}`,
      });
    }
    if (cats.length > 0) {
      prompts.push({
        label: `Compare ${kpi} by ${cats[0]}`,
        prompt: `Compare ${kpi} across ${cats[0]} with a breakdown`,
      });
    }
    if (nums.length >= 2) {
      prompts.push({
        label: `Correlate ${nums[0]} vs ${nums[1]}`,
        prompt: `Analyze the correlation between ${nums[0]} and ${nums[1]}`,
      });
    }
    if (cats.length >= 2) {
      prompts.push({
        label: `Segment ${kpi} by ${cats[0]} & ${cats[1]}`,
        prompt: `Show ${kpi} segmented by ${cats[0]} and ${cats[1]} with a stacked breakdown`,
      });
    }

    return prompts.slice(0, 3);
  };

  const processDatasetResult = (res, filename, { restore = false } = {}) => {
    if (!res) return;
    const summaryObj = res.summary || {};
    setCurrentFileName(
      filename || res.filename || summaryObj.filename || "dataset",
    );
    setDatasetSummary(summaryObj);
    setDatasetId(res.dataset_id);
    activeIdRef.current = res.dataset_id;
    localStorage.setItem(ACTIVE_KEY, res.dataset_id);
    setForecast(res.forecast || null);
    if (res.plan_source) setPlanSource(res.plan_source);

    if (summaryObj.available_kpi_options) {
      setAvailableKpiOptions(summaryObj.available_kpi_options);
      setSelectedKpiColumn(summaryObj.primary_kpi);
    }

    // Normalize chart objects whether coming from Phase 1 or client adapter
    const rawCharts = res.charts || [];
    const typeLabelMap = {
      bar: "Bar Chart",
      line: "Line Graph",
      area: "Area Graph",
      pie: "Pie Chart",
      scatter: "Scatterplot",
      histogram: "Histogram",
    };

    const normalizedCharts = rawCharts.map((c, i) => {
      const chartId = c.id || `auto-${i}`;
      const mappedType =
        typeLabelMap[(c.type || c.chart_type || "").toLowerCase()] ||
        c.type ||
        c.chart_type ||
        "Bar Chart";
      const xAxis = c.x_axis || c.x_column || "";
      const yAxis = c.y_axis || c.y_column || "";
      const agg = (c.aggregation || "sum").toLowerCase();

      return {
        id: chartId,
        title: c.title || `${mappedType} Overview`,
        type: mappedType,
        x_axis: xAxis,
        y_axis: yAxis,
        secondary_dimension: c.secondary_dimension,
        aggregation: agg,
        format_type: c.format_type,
        unit: c.unit,
        matrix_data: c.matrix_data,
        data: c.data || [],
        insight_tooltip:
          c.insight_tooltip ||
          c.description ||
          `Aggregated ${yAxis} by ${xAxis}.`,
      };
    });

    if (normalizedCharts.length > 0) {
      const initialConfigs = {};
      normalizedCharts.forEach((c) => {
        initialConfigs[c.id] = {
          x_axis: c.x_axis,
          y_axis: c.y_axis,
          secondary_dimension: c.secondary_dimension,
          aggregation: c.aggregation || "sum",
          format_type: c.format_type,
          unit: c.unit,
        };
      });
      setChartConfigs(initialConfigs);
      setActiveCharts(normalizedCharts);
    }

    setDashboardTitle(
      res.dashboard_title ||
        `${(filename || "Dataset").replace(/\.[^/.]+$/, "")} — Analytics Dashboard`,
    );

    // Fallback logic so PRIMARY KPI never displays "—"
    const firstKpiFromArray =
      Array.isArray(res.kpis) && res.kpis.length > 0 ? res.kpis[0] : null;
    const firstChart = normalizedCharts[0];
    const fallbackChartSum =
      firstChart?.data?.length > 0
        ? firstChart.data.reduce(
            (acc, item) => acc + (Number(item.value) || 0),
            0,
          )
        : null;

    const resolvedKpiName =
      res.kpi_summary?.primary_kpi ||
      firstKpiFromArray?.label ||
      summaryObj.primary_kpi ||
      firstChart?.y_axis ||
      "Primary Metric";

    let rawKpiVal =
      res.kpi_summary?.formatted_value ||
      res.kpi_summary?.value ||
      firstKpiFromArray?.value ||
      fallbackChartSum;

    let formattedKpiVal = "—";
    if (typeof rawKpiVal === "number" && !Number.isNaN(rawKpiVal)) {
      formattedKpiVal =
        Math.abs(rawKpiVal) >= 1_000_000
          ? `${(rawKpiVal / 1_000_000).toFixed(2)}M`
          : Math.abs(rawKpiVal) >= 1_000
            ? `${(rawKpiVal / 1_000).toFixed(1)}K`
            : rawKpiVal.toLocaleString();
    } else if (
      typeof rawKpiVal === "string" &&
      rawKpiVal.trim() !== "" &&
      rawKpiVal !== "—"
    ) {
      formattedKpiVal = rawKpiVal;
    }

    const rowCount =
      res.kpi_summary?.total_rows ||
      summaryObj.row_count ||
      res.cleaning_report?.cleaned_rows ||
      0;

    setKpiSummary({
      primary_kpi: resolvedKpiName,
      value: formattedKpiVal,
      formatted_value: formattedKpiVal,
      secondary_kpi:
        res.kpi_summary?.secondary_kpi || (res.kpis?.[1]?.label ?? null),
      secondary_value:
        res.kpi_summary?.secondary_formatted_value ||
        (typeof res.kpis?.[1]?.value === "number"
          ? res.kpis[1].value.toLocaleString()
          : null),
      data_quality:
        res.kpi_summary?.data_quality || summaryObj.quality_score || 100.0,
      total_rows: Number(rowCount).toLocaleString(),
    });

    setDetailedReport(
      res.executive_summary
        ? `# Executive Summary\n\n${res.executive_summary}\n\n## Key Observations\n${(res.key_observations || []).map((o) => `- ${o}`).join("\n")}`
        : "Initial analysis complete. Ask any analytical query to generate an executive report.",
    );
    if (!restore) refreshDatasets();
  };

  // Handler for Primary KPI dropdown change
  const handleKpiChange = async (newColumn) => {
    if (!datasetId || !newColumn || newColumn === selectedKpiColumn) return;

    const option = availableKpiOptions.find((o) => o.column === newColumn);
    const overrides = option
      ? {
          aggregation: option.aggregation,
          format_type: option.format_type,
          unit: option.unit,
        }
      : {};

    setSelectedKpiColumn(newColumn);

    try {
      const res = await selectPrimaryKpi(datasetId, newColumn, overrides);
      setKpiSummary((prev) => ({
        ...prev,
        primary_kpi: res.primary_kpi,
        value: res.value,
        formatted_value: res.formatted_value,
        aggregation: res.aggregation,
      }));

      // Update chart configs for charts that use the old primary KPI as Y-axis
      setChartConfigs((prev) => {
        const updated = { ...prev };
        Object.keys(updated).forEach((chartId) => {
          if (updated[chartId].y_axis === selectedKpiColumn) {
            updated[chartId] = { ...updated[chartId], y_axis: newColumn };
          }
        });
        return updated;
      });

      // Re-aggregate dependent charts
      if (activeCharts.length > 0) {
        for (const chart of activeCharts) {
          if (
            chart.y_axis === selectedKpiColumn ||
            chart.y_axis === kpiSummary.primary_kpi
          ) {
            try {
              await handleAxisChange(chart.id, { y_axis: newColumn });
            } catch (e) {
              console.warn(`Failed to re-aggregate chart ${chart.id}:`, e);
            }
          }
        }
      }
    } catch (err) {
      console.error("KPI change failed:", err);
      setSelectedKpiColumn(selectedKpiColumn); // Rollback
    }
  };

  // Handler for chart axis/aggregation change
  const handleAxisChange = async (chartId, changes) => {
    if (!datasetId) return;

    const chart = activeCharts.find((c) => c.id === chartId);
    if (!chart) return;

    const currentConfig = chartConfigs[chartId] || {};
    const newConfig = { ...currentConfig, ...changes };

    // Optimistically update config
    setChartConfigs((prev) => ({ ...prev, [chartId]: newConfig }));

    try {
      const res = await aggregateChart(datasetId, {
        chart_id: chartId,
        type: chart.type,
        x_axis: newConfig.x_axis,
        y_axis: newConfig.y_axis,
        secondary_dimension: newConfig.secondary_dimension,
        aggregation: newConfig.aggregation,
        format_type: newConfig.format_type,
        unit: newConfig.unit,
        title: chart.title,
      });

      // Update chart with new data
      setActiveCharts((prev) =>
        prev.map((c) => (c.id === chartId ? { ...c, ...res } : c)),
      );

      // If KPI changed, update KPI summary
      if (changes.y_axis && res.y_axis) {
        const option = availableKpiOptions.find((o) => o.column === res.y_axis);
        if (option) {
          // Update KPI summary if this is the primary chart
          const primaryChart = activeCharts[0];
          if (primaryChart && primaryChart.id === chartId) {
            setKpiSummary((prev) => ({
              ...prev,
              primary_kpi: res.y_axis,
              value: res.data?.[0]?.value || prev.value,
              formatted_value:
                res.data?.[0]?.formatted_value || prev.formatted_value,
            }));
          }
        }
      }
    } catch (err) {
      console.error("Chart aggregation failed:", err);
      // Rollback config
      setChartConfigs((prev) => ({ ...prev, [chartId]: currentConfig }));
    }
  };

  const handleDownloadData = async () => {
    if (!datasetId) return;
    setIsDownloading(true);
    try {
      const fallback = currentFileName
        ? `${currentFileName.replace(/\.[^/.]+$/, "")}_cleaned.csv`
        : "cleaned_dataset.csv";
      await downloadDataset(datasetId, fallback);
      setChatHistory((prev) => [
        ...prev,
        {
          role: "ai",
          text: "📥 Successfully exported and downloaded cleaned dataset in CSV format.",
        },
      ]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: "ai",
          text: `Download Error: ${err.message}`,
          type: "error",
        },
      ]);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleExportPDF = async () => {
    if (!datasetId) return;
    setIsDownloading(true);
    try {
      const fallback = currentFileName
        ? `${currentFileName.replace(/\.[^/.]+$/, "")}_executive_report.pdf`
        : "executive_report.pdf";
      await exportReportPDF(
        datasetId,
        chatInput || undefined,
        detailedReport,
        fallback,
        activeCharts,
      );
      setChatHistory((prev) => [
        ...prev,
        {
          role: "ai",
          text: "📄 Successfully exported executive report as PDF.",
        },
      ]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: "ai",
          text: `PDF Export Error: ${err.message}`,
          type: "error",
        },
      ]);
    } finally {
      setIsDownloading(false);
    }
  };

  const preprocessKeywords = [
    "add a column",
    "add column",
    "drop column",
    "remove column",
    "rename column",
    "fill missing",
    "fill in missing",
    "change type",
    "convert column",
    "filter rows",
    "filter out",
  ];

  const isPreprocessCommand = (msg) => {
    const lower = msg.toLowerCase();
    return preprocessKeywords.some((kw) => lower.includes(kw));
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = chatInput;
    setChatInput("");
    addMessage({ role: "user", text: userMsg });

    if (!datasetId) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: "ai",
          text: "Please upload a CSV or Excel dataset to begin your analysis and ask analytical queries.",
        },
      ]);
      return;
    }

    if (isPreprocessCommand(userMsg)) {
      setChatHistory((prev) => [
        ...prev,
        { role: "ai", text: "⚙️ Executing dataset preprocessing..." },
      ]);
      try {
        const res = await preprocessDataset(datasetId, userMsg);
        const confirmation =
          res.confirmation_message || "Preprocessing completed.";
        setChatHistory((prev) => {
          const updated = [...prev];
          if (updated[updated.length - 1]?.role === "ai") {
            updated[updated.length - 1] = { role: "ai", text: confirmation };
          }
          return updated;
        });
        saveMessage(activeIdRef.current, "ai", confirmation).catch(() => {});
        if (activeView === "datasets") {
          await fetchPreview(previewData.page);
        }
      } catch (err) {
        setChatHistory((prev) => {
          const updated = [...prev];
          if (updated[updated.length - 1]?.role === "ai") {
            updated[updated.length - 1] = {
              role: "ai",
              text: err.message || "Preprocessing failed.",
              type: "error",
            };
          }
          return updated;
        });
      }
      return;
    }

    // Normal analytical query path with live reasoning
    setIsThinking(true);
    setActiveReasoningText("");

    let accumulatedReasoning = "";

    await streamAIQuery(
      userMsg,
      datasetId,
      (thinkingChunk) => {
        accumulatedReasoning += thinkingChunk;
        setActiveReasoningText(accumulatedReasoning);
      },
      (payload) => {
        setIsThinking(false);
        setActiveReasoningText("");
        if (!payload) return;

        if (payload.dashboard_title) {
          setDashboardTitle(payload.dashboard_title);
        }

        // Normalize & add any new charts returned by the chatbot
        const rawCharts =
          payload.suggested_charts || (payload.chart ? [payload.chart] : []);
        if (rawCharts.length > 0) {
          const typeLabelMap = {
            bar: "Bar Chart",
            line: "Line Graph",
            area: "Area Graph",
            pie: "Pie Chart",
            scatter: "Scatterplot",
            histogram: "Histogram",
          };

          const incomingCharts = rawCharts.map((c, i) => {
            const chartId = c.id || `stream-${Date.now()}-${i}`;
            const mappedType =
              typeLabelMap[(c.type || c.chart_type || "").toLowerCase()] ||
              c.type ||
              c.chart_type ||
              "Bar Chart";
            const xAxis = c.x_axis || c.x_column || "";
            const yAxis = c.y_axis || c.y_column || "";
            const agg = (c.aggregation || "sum").toLowerCase();

            return {
              id: chartId,
              title: c.title || `${mappedType} Chart`,
              type: mappedType,
              x_axis: xAxis,
              y_axis: yAxis,
              aggregation: agg,
              secondary_dimension: c.secondary_dimension,
              matrix_data: c.matrix_data,
              data: c.data || [],
              insight_tooltip:
                c.insight_tooltip ||
                c.description ||
                "Verified calculation from cleaned dataset.",
            };
          });

          // Register axis/aggregation dropdown configs for the new chart(s)
          setChartConfigs((prev) => {
            const next = { ...prev };
            incomingCharts.forEach((c) => {
              next[c.id] = {
                x_axis: c.x_axis,
                y_axis: c.y_axis,
                secondary_dimension: c.secondary_dimension,
                aggregation: c.aggregation || "sum",
              };
            });
            return next;
          });

          // If full dashboard regeneration (4+ charts), replace; if 1-3 custom chat charts, prepend to top
          setActiveCharts((prev) =>
            incomingCharts.length >= 4
              ? incomingCharts
              : [...incomingCharts, ...prev],
          );
        }

        if (payload.kpi_summary) {
          setKpiSummary((prev) => ({
            ...prev,
            primary_kpi: payload.kpi_summary.primary_kpi || prev.primary_kpi,
            value:
              payload.kpi_summary.formatted_value ||
              (typeof payload.kpi_summary.value === "number"
                ? payload.kpi_summary.value.toLocaleString()
                : prev.value),
            total_rows:
              payload.kpi_summary.total_rows?.toLocaleString?.() ||
              prev.total_rows,
            data_quality: payload.kpi_summary.data_quality ?? prev.data_quality,
          }));
        }

        if (payload.detailed_report) setDetailedReport(payload.detailed_report);
        if (payload.forecast) setForecast(payload.forecast);
        if (payload.plan_source) setPlanSource(payload.plan_source);

        const sourceLabel =
          payload.plan_source === "llm"
            ? "✨ Nemotron-3 Ultra 550B"
            : "⚙️ Rule-based fallback";

        const replyText =
          payload.answer ||
          (rawCharts.length > 0
            ? `✨ Added ${rawCharts.length} visualization(s) for: "${userMsg}".`
            : `✨ Analysis complete for: "${userMsg}".`);

        addMessage({
          role: "ai",
          text: replyText,
          sourceLabel: sourceLabel,
          fullReasoning: accumulatedReasoning,
        });
      },
      (error) => {
        setIsThinking(false);
        setActiveReasoningText("");
        const errText =
          typeof error === "string"
            ? error
            : Array.isArray(error)
              ? error.map((e) => `${e.loc?.join(".")}: ${e.msg}`).join(", ")
              : error?.detail
                ? typeof error.detail === "string"
                  ? error.detail
                  : JSON.stringify(error.detail)
                : JSON.stringify(error);

        setChatHistory((prev) => [
          ...prev,
          {
            role: "ai",
            text: `Query Error: ${errText}`,
            type: "error",
          },
        ]);
      },
    );
  };

  const handlePromptClick = (prompt) => {
    setChatInput(prompt);
  };

  const fetchPreview = async (pageNum = 1) => {
    if (!datasetId) return;
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const data = await fetchDatasetPreview(datasetId, pageNum, 50);
      setPreviewData(data);
    } catch (err) {
      setPreviewError(err.message || "Failed to load preview");
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleViewChange = (view) => {
    setActiveView(view);
    if (view === "datasets" && datasetId) {
      fetchPreview(1);
    }
  };

  return (
    <div className="dashboard-layout">
      {(isThinking || isUploading || isDownloading || isRestoring) && (
        <div className="top-shimmer-bar" />
      )}

      <AddChartModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAddChart={handleAddChart}
      />
      <ReportModal
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
        report={detailedReport}
        datasetName={currentFileName}
      />

      {/* Left Sidebar */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-top">
          <button className="icon-btn" onClick={onBack} title="Back to Home">
            <ChevronLeft size={20} />
          </button>
          <div className="divider"></div>
          <button
            className={`icon-btn ${activeView === "dashboard" ? "active" : ""}`}
            title="Dashboard"
            onClick={() => handleViewChange("dashboard")}
          >
            <LayoutDashboard size={20} />
          </button>
          <button
            className={`icon-btn ${activeView === "datasets" ? "active" : ""}`}
            title="Datasets Preview"
            onClick={() => handleViewChange("datasets")}
          >
            <Database size={20} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <button
            className="icon-btn"
            title="Toggle Theme"
            onClick={toggleTheme}
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="workspace">
        {activeView === "dashboard" ? (
          <>
            <header className="workspace-header">
              <div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "14px",
                    flexWrap: "wrap",
                  }}
                >
                  <h2>{dashboardTitle}</h2>
                  {(isThinking || isUploading) && (
                    <div className="ai-active-pill glass-panel">
                      <div className="ai-pulse-dot" />
                      <span className="ai-active-title">
                        {isUploading ? "Ingesting Dataset" : "Nemotron-3 Ultra"}
                      </span>
                      <span className="ai-active-step">
                        {isUploading
                          ? (runningStep?.label ?? "Preparing")
                          : activeReasoningText
                            ? "Streaming reasoning & aggregations"
                            : "Analyzing intent & selecting visuals"}
                      </span>
                      <span className="ai-active-timer">
                        <Clock size={12} /> {elapsedTime.toFixed(1)}s
                      </span>
                    </div>
                  )}
                </div>
                <div
                  style={{
                    fontSize: "0.85rem",
                    color: "var(--c-text-secondary)",
                    marginTop: "4px",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  <span>{currentFileName || "No dataset loaded"}</span>
                  {currentFileName && (
                    <span
                      style={{
                        fontSize: "0.72rem",
                        padding: "2px 8px",
                        borderRadius: "999px",
                        background: "rgba(139,92,246,0.15)",
                        color: "var(--c-accent-purple)",
                        border: "1px solid rgba(139,92,246,0.3)",
                        fontWeight: 600,
                      }}
                    >
                      ✨ Nemotron-3 Ultra 550B
                    </span>
                  )}
                </div>
              </div>

              <div className="header-actions">
                <DatasetPicker
                  datasets={datasets}
                  activeId={datasetId || pendingDatasetId}
                  onOpen={openWorkspace}
                  onDelete={handleDeleteDataset}
                  disabled={isUploading || isThinking}
                />
                <label
                  className="btn btn-glass"
                  style={{
                    cursor:
                      isUploading || isThinking ? "not-allowed" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    opacity: isUploading || isThinking ? 0.7 : 1,
                  }}
                >
                  <Upload size={16} />{" "}
                  {isUploading ? "Uploading..." : "Upload CSV/Excel"}
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileUpload}
                    style={{ display: "none" }}
                    disabled={isUploading || isThinking}
                  />
                </label>

                <button
                  className="btn btn-glass"
                  onClick={() => setIsReportOpen(true)}
                  style={{ gap: "6px" }}
                  disabled={!datasetId}
                >
                  <FileText size={16} /> Executive Report
                </button>

                <button
                  className="btn btn-glass"
                  onClick={handleExportPDF}
                  style={{ gap: "6px" }}
                  disabled={!datasetId || isDownloading}
                >
                  {isDownloading ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <FileText size={16} />
                  )}
                  {isDownloading ? "Generating PDF..." : "Export PDF"}
                </button>

                <button
                  className="btn btn-primary"
                  onClick={() => setIsAddModalOpen(true)}
                  style={{ gap: "6px" }}
                  disabled={!datasetId}
                >
                  <Plus size={16} /> Add Chart
                </button>
              </div>
            </header>

            {/* Top KPI Cards Row */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "12px",
                marginBottom: "14px",
                flexShrink: 0,
              }}
            >
              <div
                className="glass-panel kpi-card"
                style={{
                  padding: "12px 16px",
                  borderRadius: "14px",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: "4px",
                  }}
                >
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--c-text-secondary)",
                      fontWeight: 600,
                    }}
                  >
                    PRIMARY KPI
                  </span>
                  {availableKpiOptions.length > 1 && (
                    <select
                      className="kpi-dropdown glass-panel"
                      value={selectedKpiColumn || kpiSummary.primary_kpi}
                      onChange={(e) => handleKpiChange(e.target.value)}
                      disabled={isThinking || isUploading}
                      style={{
                        padding: "4px 8px",
                        borderRadius: "8px",
                        border: "1px solid var(--c-glass-border)",
                        background: "var(--c-glass-bg)",
                        color: "var(--c-text)",
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        cursor:
                          isThinking || isUploading ? "not-allowed" : "pointer",
                        opacity: isThinking || isUploading ? 0.6 : 1,
                        minWidth: "120px",
                      }}
                    >
                      {availableKpiOptions.map((opt) => (
                        <option key={opt.column} value={opt.column}>
                          {opt.display_name} ({opt.unit || opt.format_type})
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <div
                  style={{
                    fontSize: "1.4rem",
                    fontWeight: 700,
                    marginTop: "2px",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    color: "var(--c-accent-blue)",
                  }}
                >
                  {kpiSummary.formatted_value || kpiSummary.value}
                </div>
              </div>

              {kpiSummary.secondary_kpi ? (
                <div
                  className="glass-panel"
                  style={{ padding: "12px 16px", borderRadius: "14px" }}
                >
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--c-text-secondary)",
                      fontWeight: 600,
                    }}
                  >
                    SECONDARY KPI ({kpiSummary.secondary_kpi})
                  </span>
                  <div
                    style={{
                      fontSize: "1.4rem",
                      fontWeight: 700,
                      marginTop: "2px",
                      color: "var(--c-accent-emerald)",
                    }}
                  >
                    {kpiSummary.secondary_value || "—"}
                  </div>
                </div>
              ) : (
                <div
                  className="glass-panel"
                  style={{ padding: "12px 16px", borderRadius: "14px" }}
                >
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--c-text-secondary)",
                      fontWeight: 600,
                    }}
                  >
                    CLEANED RECORDS
                  </span>
                  <div
                    style={{
                      fontSize: "1.4rem",
                      fontWeight: 700,
                      marginTop: "2px",
                    }}
                  >
                    {kpiSummary.total_rows}
                  </div>
                </div>
              )}

              <div
                className="glass-panel"
                style={{ padding: "12px 16px", borderRadius: "14px" }}
              >
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--c-text-secondary)",
                    fontWeight: 600,
                  }}
                >
                  {kpiSummary.secondary_kpi
                    ? "CLEANED RECORDS"
                    : "DATA QUALITY SCORE"}
                </span>
                <div
                  style={{
                    fontSize: "1.4rem",
                    fontWeight: 700,
                    marginTop: "2px",
                    color: kpiSummary.secondary_kpi
                      ? "var(--c-text)"
                      : "var(--c-accent-emerald)",
                  }}
                >
                  {kpiSummary.secondary_kpi
                    ? kpiSummary.total_rows
                    : `${kpiSummary.data_quality}%`}
                </div>
              </div>

              <div
                className="glass-panel"
                style={{ padding: "12px 16px", borderRadius: "14px" }}
              >
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--c-text-secondary)",
                    fontWeight: 600,
                  }}
                >
                  {forecast?.projected_growth_rate
                    ? "PROJECTED GROWTH RATE"
                    : "DATA INTEGRITY"}
                </span>
                <div
                  style={{
                    fontSize: "1.4rem",
                    fontWeight: 700,
                    marginTop: "2px",
                    color: "var(--c-accent-purple)",
                  }}
                >
                  {forecast?.projected_growth_rate ||
                    `${kpiSummary.data_quality}%`}
                </div>
              </div>
            </div>

            {/* Canvas & Charts */}
            <div className="canvas">
              {isThinking && (
                <div className="canvas-processing-banner glass-panel">
                  <div className="banner-left">
                    <Sparkles
                      size={18}
                      color="var(--c-accent-purple)"
                      className="animate-spin-slow"
                    />
                    <div>
                      <div className="banner-text">
                        Nemotron-3 Ultra 550B is formulating visual
                        intelligence...
                      </div>
                      <div className="banner-subtext">
                        {activeReasoningText
                          ? "Streaming live thoughts, verifying column axes, and calculating distributions..."
                          : "Parsing prompt intent and selecting optimal chart compositions..."}
                      </div>
                    </div>
                  </div>
                  <div className="ai-active-timer">
                    <Clock size={12} /> {elapsedTime.toFixed(1)}s
                  </div>
                </div>
              )}
              <div className="charts-grid">
                {activeCharts.map((chart) => (
                  <ChartCard
                    key={chart.id}
                    chart={chart}
                    isExpanded={expandedCard === chart.id}
                    onToggleExpand={handleToggleExpand}
                    onRemove={handleRemoveChart}
                    onTypeChange={handleTypeChange}
                    chartConfig={chartConfigs[chart.id]}
                    onAxisChange={handleAxisChange}
                    availableKpiOptions={availableKpiOptions}
                    datasetSummary={datasetSummary}
                    isThinking={isThinking}
                  />
                ))}

                {!activeCharts.length && !isUploading && (
                  <div
                    className="recommendation-card liquid-glass"
                    style={{
                      gridColumn: "1 / -1",
                      padding: "32px",
                      textAlign: "center",
                    }}
                  >
                    <Database
                      size={36}
                      color="var(--c-accent-purple)"
                      style={{ margin: "0 auto 12px" }}
                    />
                    <h3 style={{ fontSize: "1.2rem", marginBottom: "8px" }}>
                      Ready to Analyze Your Data
                    </h3>
                    <p
                      style={{
                        fontSize: "0.9rem",
                        color: "var(--c-text-secondary)",
                        maxWidth: "480px",
                        margin: "0 auto",
                      }}
                    >
                      Upload any CSV or Excel file to begin. The AI analytics
                      engine will automatically execute automated EDA, data
                      cleaning, and intelligent visual dashboard generation.
                    </p>
                  </div>
                )}
              </div>

              {/* Executive Insights & Recommendations Section */}
              {/* Spacer to guarantee scroll visibility */}
              <div style={{ gridColumn: "1 / -1", height: "32px" }} />
            </div>
          </>
        ) : (
          <div
            style={{
              height: "calc(100% - 20px)",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <header className="workspace-header" style={{ flexShrink: 0 }}>
              <div>
                <h2>Dataset Preview & Preprocessing</h2>
                <div
                  style={{
                    fontSize: "0.85rem",
                    color: "var(--c-text-secondary)",
                    marginTop: "2px",
                  }}
                >
                  Cleaned & preprocessed table view{" "}
                  {currentFileName ? `• ${currentFileName}` : ""}
                </div>
              </div>
              <div
                style={{ display: "flex", alignItems: "center", gap: "12px" }}
              >
                <button
                  className="btn btn-primary"
                  onClick={handleDownloadData}
                  disabled={!datasetId || isDownloading}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "8px",
                  }}
                  title="Download full preprocessed dataset in CSV format"
                >
                  <Download size={16} />
                  {isDownloading ? "Downloading..." : "Download Cleaned CSV"}
                </button>
              </div>
            </header>

            <div
              className="glass-panel"
              style={{
                borderRadius: "16px",
                overflow: "hidden",
                flex: 1,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {previewLoading ? (
                <div
                  style={{
                    padding: "40px",
                    textAlign: "center",
                    color: "var(--c-text-secondary)",
                  }}
                >
                  Loading records...
                </div>
              ) : previewError ? (
                <div
                  style={{ padding: "24px", color: "var(--c-accent-orange)" }}
                >
                  Error: {previewError}
                </div>
              ) : previewData.columns.length > 0 ? (
                <>
                  <div style={{ overflowX: "auto", flex: 1 }}>
                    <table
                      style={{
                        width: "100%",
                        borderCollapse: "collapse",
                        fontSize: "0.85rem",
                      }}
                    >
                      <thead style={{ position: "sticky", top: 0, zIndex: 1 }}>
                        <tr
                          style={{
                            background: "var(--c-glass-bg)",
                            borderBottom: "1px solid var(--c-glass-border)",
                          }}
                        >
                          {previewData.columns.map((col) => (
                            <th
                              key={col}
                              style={{
                                padding: "10px 14px",
                                textAlign: "left",
                                fontWeight: 600,
                                color: "var(--c-text)",
                                whiteSpace: "nowrap",
                              }}
                            >
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.rows.map((row, i) => (
                          <tr
                            key={i}
                            style={{
                              borderBottom: "1px solid var(--c-glass-border)",
                              background:
                                i % 2 === 0
                                  ? "transparent"
                                  : "var(--c-glass-bg)",
                            }}
                          >
                            {previewData.columns.map((col) => (
                              <td
                                key={col}
                                style={{
                                  padding: "10px 14px",
                                  color: "var(--c-text)",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {row[col] === null || row[col] === undefined ? (
                                  <span
                                    style={{ color: "var(--c-text-muted)" }}
                                  >
                                    -
                                  </span>
                                ) : (
                                  String(row[col])
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div
                    style={{
                      padding: "12px 16px",
                      borderTop: "1px solid var(--c-glass-border)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      flexWrap: "wrap",
                      gap: "12px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.85rem",
                        color: "var(--c-text-secondary)",
                      }}
                    >
                      Page {previewData.page} of {previewData.total_pages || 1}{" "}
                      — {previewData.total_rows.toLocaleString()} total rows
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                      }}
                    >
                      <button
                        className="btn btn-glass"
                        disabled={previewData.page <= 1}
                        onClick={() => fetchPreview(previewData.page - 1)}
                      >
                        Previous
                      </button>
                      <button
                        className="btn btn-glass"
                        disabled={previewData.page >= previewData.total_pages}
                        onClick={() => fetchPreview(previewData.page + 1)}
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <div
                  style={{
                    padding: "40px",
                    textAlign: "center",
                    color: "var(--c-text-secondary)",
                  }}
                >
                  No dataset loaded. Upload a file to preview.
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Right AI Copilot */}
      <aside className="copilot-panel glass-panel">
        <div className="copilot-header">
          <div className="ai-avatar">
            <Sparkles size={16} />
          </div>
          <div>
            <h3>AutoInsights Copilot</h3>
            <span
              style={{
                fontSize: "0.75rem",
                color: "var(--c-accent-emerald)",
                fontWeight: 600,
              }}
            >
              ● Nemotron-3 Ultra 550B
            </span>
          </div>
        </div>

        <div className="chat-history">
          {chatHistory.map((msg, idx) => (
            <div
              key={idx}
              className={`chat-bubble ${msg.role}`}
              style={
                msg.type === "error"
                  ? {
                      borderLeft: "4px solid var(--c-accent-orange)",
                      background: "rgba(255,165,0,0.15)",
                    }
                  : {}
              }
            >
              {msg.role === "ai" ? (
                <MarkdownRenderer content={msg.text} />
              ) : (
                <div style={{ whiteSpace: "pre-wrap" }}>{msg.text}</div>
              )}
              {msg.role === "ai" && msg.sourceLabel && (
                <div style={{ marginTop: "6px" }}>
                  <span
                    style={{
                      fontSize: "0.7rem",
                      padding: "2px 7px",
                      borderRadius: "999px",
                      background: "rgba(139,92,246,0.15)",
                      color: "var(--c-accent-purple)",
                      border: "1px solid rgba(139,92,246,0.3)",
                      fontWeight: 600,
                    }}
                  >
                    {msg.sourceLabel}
                  </span>
                </div>
              )}
            </div>
          ))}

          {/* Real-Time Live Reasoning Box when isThinking is active */}
          {isThinking && (
            <div className="live-reasoning-card animate-fade-in-up">
              <div className="reasoning-header">
                <div className="reasoning-title-group">
                  <div className="ai-pulse-dot" />
                  <span>Nemotron-3 Ultra 550B Active</span>
                </div>
                <span className="ai-active-timer">
                  <Clock size={11} /> {elapsedTime.toFixed(1)}s
                </span>
              </div>
              <div className="terminal-reasoning-stream">
                {activeReasoningText ||
                  "Connecting to NVIDIA Ultra 550B inference gateway & synthesizing schema..."}
                <span className="cursor-blink" />
              </div>
              <div className="step-chips">
                <span className="step-chip done">
                  <Check size={10} /> Schema Ready
                </span>
                <span
                  className={`step-chip ${activeReasoningText ? "active" : "pending"}`}
                >
                  <Activity size={10} />{" "}
                  {activeReasoningText
                    ? "Ultra 550B Reasoning"
                    : "Synthesizing"}
                </span>
                <span className="step-chip pending">
                  <BarChart2 size={10} /> Materializing Visuals
                </span>
              </div>
            </div>
          )}

          {(isUploading || pipelineError) && (
            <PipelineProgress
              steps={pipelineSteps}
              error={pipelineError}
              canRetry={!!pendingDatasetId}
              onRetry={handleRetry}
            />
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="suggested-prompts">
          {getSuggestedPrompts().map((sp, idx) =>
            sp.prompt ? (
              <button key={idx} onClick={() => handlePromptClick(sp.prompt)}>
                {sp.label}
              </button>
            ) : (
              <button
                key={idx}
                disabled
                style={{ opacity: 0.5, cursor: "default" }}
              >
                {sp.label}
              </button>
            ),
          )}
        </div>

        <form className="chat-input-area" onSubmit={handleChatSubmit}>
          <input
            type="text"
            placeholder="Ask any question about your data..."
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            disabled={isThinking}
          />
          <button type="submit" className="send-btn" disabled={isThinking}>
            <Sparkles size={16} />
          </button>
        </form>
      </aside>
    </div>
  );
};

export default Dashboard;
