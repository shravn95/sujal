# AutoInsight Analytics

Upload a CSV or Excel dataset, then ask a question in plain language. AutoInsight cleans and profiles the data locally, computes the chart series locally, and renders an interactive dashboard from those real values.

## What works

- CSV and Excel upload with duplicate removal and missing-value treatment
- Data profiling: schema, date/dimension/KPI detection, correlations, and quality score
- Data-backed initial dashboard (trend, category breakdown, composition, and correlation where applicable)
- Query-driven dashboard updates, tooltips, detailed report, and local forecast
- NVIDIA Nemotron-3 Ultra 550B planner for chart selection, narrative report generation, and query-intent resolution. It receives only compact schema metadata, not the uploaded file or raw rows; all values displayed in charts remain locally computed and field-validated.
- Health endpoint (`GET /api/health`) and in-app AI Engine status badge so you always know whether the LLM is reachable.

## Run locally

1. Install backend packages:

   ```powershell
   python -m pip install -r backend/requirements.txt
   ```

2. Copy `backend/.env.example` to `backend/.env` and fill in your NVIDIA API key:

   ```powershell
   copy backend\.env.example backend\.env
   # Then edit backend/.env and replace "your_nvidia_api_key_here" with your real key
   ```

   Without a key the deterministic local planner remains fully functional (the app runs in rule-based fallback mode and shows an amber status badge).

3. In one terminal start the API:

   ```powershell
   python backend/app/main.py
   ```

4. In another terminal start the frontend:

   ```powershell
   npm install
   npm run dev
   ```

Open the Vite URL displayed in the terminal, then use **Sample CSV** or upload your own file.

## Production deployment

- Set `VITE_API_BASE_URL` in a root-level `.env` file (or your CI/CD environment) to the deployed backend URL:

  ```
  VITE_API_BASE_URL=https://your-backend.example.com/api
  ```

  See `.env.example` at the project root for the template.

- Dataset state is held in application memory for this local single-user deliverable. Before a multi-user/cloud deployment, move uploaded data to encrypted object storage and persist dataset metadata/session ownership in a database.

## Security — API key handling

> **NEVER put a real API key in `backend/.env.example`.**
>
> `.env.example` is committed to source control and is public. It must contain only a placeholder (`your_nvidia_api_key_here`). The real key lives exclusively in `backend/.env`, which is listed in `.gitignore` and must never be committed.
>
> If you accidentally commit a real key to `.env.example` or any other tracked file, rotate it immediately at [build.nvidia.com](https://build.nvidia.com) — treat the exposed key as compromised regardless of whether you have since deleted it, because git history preserves old file contents.
