// Configurable via VITE_API_BASE_URL in a root .env file for production deployments.
const BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

// ---------------------------------------------------------------------------
// Timeout helper — creates an AbortSignal that fires after `ms` milliseconds
// ---------------------------------------------------------------------------
const withTimeout = (ms) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(id) };
};

// ---------------------------------------------------------------------------
// Upload Dataset — 90s timeout, explicit error messages
// ---------------------------------------------------------------------------
export const STEPS = [
  { id: "upload", label: "Uploading & cleaning data" },
  { id: "analyze", label: "AI analysing dataset" },
  { id: "dashboard", label: "Building charts & KPIs" },
];

const postStep = async (path, { body, timeoutMs, label }) => {
  const { signal, clear } = withTimeout(timeoutMs);
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      body,
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `${label} failed (HTTP ${res.status})`);
    }
    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`${label} timed out after ${timeoutMs / 1000}s.`);
    }
    throw err;
  } finally {
    clear();
  }
};

const runStep = async (onStep, id, fn) => {
  onStep(id, "running");
  try {
    const result = await fn();
    onStep(id, "done", result);
    return result;
  } catch (err) {
    onStep(id, "error", err.message);
    throw err;
  }
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const getJson = async (path, label) => {
  const { signal, clear } = withTimeout(15_000);
  try {
    const res = await fetch(`${BASE_URL}${path}`, { signal });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `${label} failed (HTTP ${res.status})`);
    }
    return await res.json();
  } finally {
    clear();
  }
};

const ANALYZE_MAX_WAIT_MS = 10 * 60_000; // covers all retry attempts
const QUEUE_STUCK_MS = 20_000; // still 'queued' after this = Inngest isn't picking it up

const analyzeStep = async (
  id,
  onStep = () => {},
  { alreadyStarted = false } = {},
) => {
  if (!alreadyStarted) {
    await postStep(`/dataset/${id}/analyze`, {
      timeoutMs: 15_000,
      label: "Starting AI analysis",
    });
  }

  const startedAt = Date.now();
  let lastDetail = "";
  while (true) {
    await sleep(2000);
    const s = await getJson(`/dataset/${id}/status`, "AI analysis status");

    if (s.state === "done") return s; // has executive_summary, key_observations, plan_source
    if (s.state === "failed")
      throw new Error(s.message || "AI analysis failed.");

    const elapsed = Date.now() - startedAt;
    if (s.state === "queued" && elapsed > QUEUE_STUCK_MS) {
      throw new Error(
        "Analysis job was not picked up. Is the Inngest dev server running?",
      );
    }
    if (elapsed > ANALYZE_MAX_WAIT_MS)
      throw new Error("AI analysis took too long.");

    const STAGE_LABELS = {
      understanding: "Understanding the dataset…",
      planning: "Choosing the best charts…",
    };
    const detail =
      s.state === "retrying"
        ? `Attempt ${s.attempt} failed, retrying…`
        : s.attempt > 1
          ? `Attempt ${s.attempt} of ${s.max_attempts}`
          : STAGE_LABELS[s.stage] || "";
    if (detail !== lastDetail) {
      lastDetail = detail;
      onStep(id === undefined ? "analyze" : "analyze", "retrying", detail);
    }
  }
};

const dashboardStep = (id) =>
  postStep(`/dataset/${id}/dashboard`, {
    timeoutMs: 30_000,
    label: "Dashboard build",
  });

export const uploadDataset = async (file, onStep = () => {}) => {
  const formData = new FormData();
  formData.append("file", file);

  const uploaded = await runStep(onStep, "upload", () =>
    postStep("/upload", {
      body: formData,
      timeoutMs: 60_000,
      label: "Upload & cleaning",
    }),
  );
  await runStep(onStep, "analyze", () => analyzeStep(uploaded.dataset_id));
  return runStep(onStep, "dashboard", () => dashboardStep(uploaded.dataset_id));
};

// Retry from the failed step. The file is already on the server.
export const retryPipeline = async (datasetId, from, onStep = () => {}) => {
  if (from === "analyze")
    await runStep(onStep, "analyze", () => analyzeStep(datasetId));
  return runStep(onStep, "dashboard", () => dashboardStep(datasetId));
};

// ── Workspace persistence ────────────────────────────────────────────────
const request = async (
  method,
  path,
  body,
  label = "Request",
  timeoutMs = 15_000,
) => {
  const { signal, clear } = withTimeout(timeoutMs);
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers:
        body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `${label} failed (HTTP ${res.status})`);
    }
    return await res.json();
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`${label} timed out`);
    throw e;
  } finally {
    clear();
  }
};

export const listDatasets = () =>
  request("GET", "/datasets", undefined, "Loading datasets");
export const openDataset = (id) =>
  request("GET", `/datasets/${id}`, undefined, "Opening dataset", 30_000);
export const renameDataset = (id, title) =>
  request("PATCH", `/datasets/${id}`, { title }, "Rename");
export const deleteDataset = (id) =>
  request("DELETE", `/datasets/${id}`, undefined, "Delete");
export const saveMessage = (id, role, text, meta = {}) =>
  request(
    "POST",
    `/datasets/${id}/messages`,
    { role, text, meta },
    "Save message",
  );
export const clearMessages = (id) =>
  request("DELETE", `/datasets/${id}/messages`, undefined, "Clear chat");
export const saveDashboardState = (id, state) =>
  request("PUT", `/datasets/${id}/dashboard`, state, "Save dashboard", 30_000);

// Continue an interrupted pipeline from wherever it stopped
export const resumePipeline = async (datasetId, state, onStep = () => {}) => {
  if (state === "uploaded") return retryPipeline(datasetId, "analyze", onStep);
  if (state === "analyzing") {
    // The job is already running in Inngest, so poll it and don't enqueue a second one
    await runStep(onStep, "analyze", () =>
      analyzeStep(datasetId, onStep, { alreadyStarted: true }),
    );
    return runStep(onStep, "dashboard", () => dashboardStep(datasetId));
  }
  if (state === "analyzed")
    return runStep(onStep, "dashboard", () => dashboardStep(datasetId));
  throw new Error("Nothing to resume");
};

// ---------------------------------------------------------------------------
// Stream AI Query — 120s timeout, robust SSE parsing
// ---------------------------------------------------------------------------
export const streamAIQuery = async (
  query,
  datasetId,
  onThinking,
  onPayload,
  onError,
) => {
  const { signal, clear } = withTimeout(120_000);
  try {
    const response = await fetch(`${BASE_URL}/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: query,
        dataset_id: datasetId,
      }),
      signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      const detailStr =
        typeof err.detail === "string"
          ? err.detail
          : Array.isArray(err.detail)
            ? err.detail.map((e) => `${e.loc?.join(".")}: ${e.msg}`).join(", ")
            : `Query failed (HTTP ${response.status})`;
      throw new Error(detailStr);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // SSE events are separated by double newlines
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const event of events) {
        // Handle both "data: {...}" and "data:  {...}" (extra whitespace)
        const dataMatch = event.match(/^data:\s*(.+)$/ms);
        if (!dataMatch) continue;

        try {
          const data = JSON.parse(dataMatch[1].trim());
          if (data.type === "thinking") {
            onThinking && onThinking(data.content);
          } else if (data.type === "error") {
            // Structured error from backend
            onError && onError(data.content || "An unknown error occurred");
          } else if (data.type === "payload" || data.data) {
            onPayload && onPayload(data.data || data);
          }
        } catch (e) {
          // Non-JSON line — log and skip, don't crash the stream
          console.warn(
            "Skipping non-JSON SSE chunk:",
            dataMatch[1]?.substring(0, 100),
          );
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      onError &&
        onError(
          "AI query timed out after 120 seconds. Try a simpler question or check API connectivity.",
        );
    } else {
      onError && onError(err.message);
    }
  } finally {
    clear();
  }
};

// ---------------------------------------------------------------------------
// Fetch Forecast
// ---------------------------------------------------------------------------
export const fetchForecast = async (historicalValues, periods = 4) => {
  const response = await fetch(`${BASE_URL}/forecast`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ historical_values: historicalValues, periods }),
  });
  return await response.json();
};

// ---------------------------------------------------------------------------
// Dataset Preview
// ---------------------------------------------------------------------------
export const fetchDatasetPreview = async (
  datasetId,
  page = 1,
  pageSize = 50,
) => {
  const response = await fetch(
    `${BASE_URL}/dataset/${datasetId}/preview?page=${page}&page_size=${pageSize}`,
  );
  if (!response.ok) {
    const err = await response
      .json()
      .catch(() => ({ detail: "Failed to fetch dataset preview" }));
    throw new Error(err.detail);
  }
  return await response.json();
};

// ---------------------------------------------------------------------------
// Preprocess Dataset
// ---------------------------------------------------------------------------
export const preprocessDataset = async (datasetId, command) => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/preprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command }),
  });
  if (!response.ok) {
    const err = await response
      .json()
      .catch(() => ({ detail: "Preprocessing failed" }));
    throw new Error(err.detail);
  }
  return await response.json();
};

// ---------------------------------------------------------------------------
// Download Cleaned Dataset
// ---------------------------------------------------------------------------
export const downloadDataset = async (
  datasetId,
  fallbackName = "cleaned_dataset.csv",
) => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/download`);
  if (!response.ok) {
    const err = await response
      .json()
      .catch(() => ({ detail: "Failed to download dataset" }));
    throw new Error(err.detail || "Failed to download dataset");
  }

  let filename = fallbackName;
  const disposition = response.headers.get("Content-Disposition");
  if (disposition && disposition.includes("filename=")) {
    const match = disposition.match(/filename=["']?([^"']+)["']?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
};

// ---------------------------------------------------------------------------
// Health Check — AI Engine status badge
// ---------------------------------------------------------------------------
export const fetchHealth = async () => {
  try {
    const response = await fetch(`${BASE_URL}/health/llm`);
    if (!response.ok)
      return { llm_configured: false, llm_reachable: false, model: "" };
    return await response.json();
  } catch {
    return { llm_configured: false, llm_reachable: false, model: "" };
  }
};

// ---------------------------------------------------------------------------
// Export PDF Report
// ---------------------------------------------------------------------------
export const exportReportPDF = async (
  datasetId,
  query = null,
  reportText = null,
  fallbackName = "executive_report.pdf",
  charts = [],
) => {
  const response = await fetch(`${BASE_URL}/report/export-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: datasetId,
      query,
      report_text: reportText,
      charts,
    }),
  });

  if (!response.ok) {
    const err = await response
      .json()
      .catch(() => ({ detail: "Failed to export PDF report" }));
    throw new Error(err.detail || `Export failed (HTTP ${response.status})`);
  }

  let filename = fallbackName;
  const disposition = response.headers.get("Content-Disposition");
  if (disposition && disposition.includes("filename=")) {
    const match = disposition.match(/filename=["']?([^"']+)["']?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
};

// ---------------------------------------------------------------------------
// Select Primary KPI
// ---------------------------------------------------------------------------
export const selectPrimaryKpi = async (datasetId, column, overrides = {}) => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/select-kpi`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column, ...overrides }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(
      err.detail || `Select KPI failed (HTTP ${response.status})`,
    );
  }
  return await response.json();
};

// ---------------------------------------------------------------------------
// Aggregate Chart
// ---------------------------------------------------------------------------
export const aggregateChart = async (datasetId, chartSpec) => {
  const response = await fetch(
    `${BASE_URL}/dataset/${datasetId}/chart/aggregate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chartSpec),
    },
  );
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(
      err.detail || `Chart aggregation failed (HTTP ${response.status})`,
    );
  }
  return await response.json();
};
