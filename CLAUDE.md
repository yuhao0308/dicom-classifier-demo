# CLAUDE.md — AI Pair Programmer Contract & Repo Operating Manual

## 1. Project Summary

**Medical Imaging AI Demo** is a lightweight web application that accepts a DICOM CT series upload (≤100 slices), runs a classification/detection model to flag suspicious slices (e.g., pulmonary nodules or lesions), and returns annotated results: abnormal slice images with ROI bounding boxes and a one-sentence finding per slice. The stack is designed to run locally on CPU with optional GPU acceleration. This is a **demo/prototype only**.

### Non-Goals

- **Not** a diagnostic medical device — not FDA-cleared, not CE-marked, not clinically validated.
- **Not** intended for production PACS/RIS/EHR integration.
- **Not** a knowledge-base chatbot or conversational AI.
- **Not** designed for hospital-scale throughput or multi-tenant deployment.
- **Not** a replacement for radiologist interpretation.

---

## 2. Tech Stack

**Choice: Option B — Python-only (FastAPI + minimal HTML/JS)**

**Justification:** The project is a demo with a single upload-and-view workflow. A separate React/Next.js frontend adds build tooling, a second dev server, and CORS configuration — none of which accelerate a demo. FastAPI serves HTML via Jinja2 templates, static JS/CSS via `StaticFiles`, and JSON via API endpoints, all from one process. This halves setup time and keeps the repo simple. If a richer UI is needed later, a frontend app can be added without changing the API layer.

---

## 3. Repository Layout

```
dicom-classifier-demo/
├── CLAUDE.md                # This file — repo contract & operating manual
├── README.md                # User-facing overview (short)
├── pyproject.toml            # Project metadata, dependencies, tool config
├── .gitignore
├── .env.example              # Template for environment variables
│
├── app/                      # FastAPI backend application
│   ├── __init__.py
│   ├── main.py               # FastAPI app factory, lifespan, middleware
│   ├── config.py             # Settings via pydantic-settings (reads .env)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py         # POST /upload
│   │   ├── jobs.py           # GET /jobs/{id}
│   │   └── results.py        # GET /results/{id}
│   ├── services/
│   │   ├── __init__.py
│   │   ├── dicom_parser.py   # DICOM reading, validation, de-id checks
│   │   ├── inference.py      # Model loading, batch predict, heatmap
│   │   ├── postprocess.py    # Threshold heatmap → bounding box, summary
│   │   └── storage.py        # Temp file management, auto-cleanup
│   ├── templates/            # Jinja2 HTML templates
│   │   ├── index.html        # Upload page
│   │   └── results.html      # Results display page
│   └── static/               # CSS, JS, icons
│       ├── style.css
│       └── app.js
│
├── models/                   # Model weights & config (gitignored except README)
│   └── README.md             # Instructions to download/place weights
│
├── data/                     # Sample DICOMs for dev/test (entirely gitignored)
│   └── .gitkeep
│
├── docs/                     # Architecture notes, diagrams
│   └── architecture.md
│
├── scripts/                  # Dev utilities
│   ├── download_model.py     # Fetch model weights from a URL/registry
│   ├── generate_sample.py    # Create synthetic DICOM test data
│   └── clean_tempfiles.py    # Manual temp-dir cleanup
│
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures (test client, sample DICOM)
    ├── test_dicom_parser.py
    ├── test_inference.py
    ├── test_routes.py         # API smoke tests
    └── test_e2e.py            # Optional end-to-end test
```

---

## 4. Local Dev Setup

### Prerequisites

- Python 3.11+
- macOS / Linux (Windows via WSL is fine)
- `uv` (recommended) or `pip`

### Step-by-step

```bash
# 1. Clone the repo
git clone <repo-url> && cd dicom-classifier-demo

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"
# Or with uv:
# uv pip install -e ".[dev]"

# 4. Copy environment template
cp .env.example .env

# 5. (Optional) Download model weights
python scripts/download_model.py

# 6. Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 7. Open browser
# http://localhost:8000
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/classifier.pt` | Path to model weights file |
| `MAX_SLICES` | `100` | Maximum slices accepted per upload |
| `USE_GPU` | `false` | Set `true` to enable CUDA inference |
| `TEMP_DIR` | `/tmp/dicom-demo` | Directory for temporary uploaded/processed files |
| `TEMP_RETENTION_SECONDS` | `3600` | Auto-delete temp files after this many seconds |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload size in megabytes |

---

## 5. Coding Conventions & Quality Bar

### Style

- **Formatter:** `ruff format` (line length 99)
- **Linter:** `ruff check` with default rules + `I` (isort), `UP` (pyupgrade), `S` (bandit security)
- Run both before every commit: `ruff check --fix . && ruff format .`

### Type Hints

- All function signatures must have type hints (params + return).
- Use `from __future__ import annotations` at the top of every module.
- Validate runtime data with Pydantic models, not manual `isinstance` checks.

### Logging

- Use `structlog` or stdlib `logging` — pick one and stay consistent.
- **Do:** `logger.info("inference_complete", slice_count=n, elapsed_ms=t)`
- **Don't:** `print("done")` or f-string log messages (use structured key-value pairs).

### Error Handling

- FastAPI routes return proper HTTP status codes (400 for bad input, 422 for validation, 500 for unexpected).
- Never swallow exceptions silently. Log at `warning` or `error` and re-raise or return an error response.
- Use a single `app/exceptions.py` for custom exception classes if needed.

### Security — File Uploads

- Validate MIME type and DICOM magic bytes before processing.
- Enforce `MAX_UPLOAD_SIZE_MB` at the middleware level.
- Write uploads to `TEMP_DIR` only; never to the repo tree.
- Sanitize filenames — never use user-supplied filenames for filesystem paths.
- Do not serve uploaded files back to users without explicit content-type headers.

---

## 6. Data Handling Rules for Medical Images

### DICOM De-identification

- **WARNING:** DICOM files may contain Protected Health Information (PHI) in metadata tags (patient name, DOB, MRN, etc.).
- The app must **strip or ignore** PHI tags on ingest. Use `pydicom` to remove tags listed in DICOM PS3.15 Annex E before any processing or storage.
- Log a warning if common PHI tags are detected in an upload.

### No PHI in Repo

- **Never** commit real patient data, DICOM files, or metadata containing PHI.
- `data/` is gitignored entirely.
- Test fixtures must use synthetic/anonymized DICOMs only.

### Storage Retention Policy

- All uploaded and processed files go to `TEMP_DIR`.
- A background task or lifespan hook deletes files older than `TEMP_RETENTION_SECONDS`.
- On app shutdown, optionally purge all temp files.
- No database persistence in MVP — results are ephemeral.

---

## 7. Demo Workflow — Happy Path

```
User uploads DICOM .zip/.tar or folder of .dcm files
        │
        ▼
[1] VALIDATE
    - Check file count ≤ MAX_SLICES
    - Verify DICOM magic bytes
    - Strip PHI metadata tags
    - Reject non-CT modality
        │
        ▼
[2] PARSE & SAMPLE
    - Read pixel data via pydicom
    - Apply windowing (e.g., lung window W:1500 C:-600)
    - Sort by InstanceNumber / SliceLocation
    - If >MAX_SLICES, uniformly sample down
        │
        ▼
[3] INFERENCE
    - Load model (cached in memory after first call)
    - Batch predict: each slice → anomaly score + class activation map (CAM)
    - Return top-K slices above confidence threshold
        │
        ▼
[4] POSTPROCESS
    - Threshold CAM heatmap → binary mask
    - Compute bounding box from largest connected component in mask
    - Generate one-sentence finding per slice (template-based, NOT LLM-generated)
      Example: "Suspicious region detected in slice 47 (confidence: 0.82)."
        │
        ▼
[5] RENDER
    - Overlay bounding box on slice image (PNG)
    - Return results page with:
      • Annotated slice images
      • Finding text per slice
      • Prominent disclaimer banner
```

### ROI Definition (MVP)

"ROI" in this demo means a **bounding box** derived from thresholding the model's class activation map (CAM/GradCAM) at a fixed percentile (e.g., 90th). No pixel-level segmentation masks in MVP. The bounding box is the smallest axis-aligned rectangle enclosing the thresholded region.

---

## 8. Interfaces / API Contracts

### `POST /upload`

Upload a DICOM series for analysis.

**Request:** `multipart/form-data`
```
file: <binary .zip or .tar.gz containing .dcm files>
```

**Response:** `202 Accepted`
```json
{
  "job_id": "abc123",
  "status": "processing",
  "slice_count": 64,
  "message": "Upload accepted. Processing started."
}
```

**Errors:**
- `400` — No file, wrong format, exceeds MAX_SLICES
- `413` — File too large
- `422` — Not valid DICOM / not CT modality

---

### `GET /jobs/{job_id}`

Poll job status.

**Response:** `200 OK`
```json
{
  "job_id": "abc123",
  "status": "completed",
  "progress": 100,
  "created_at": "2026-02-26T10:00:00Z"
}
```

`status` is one of: `processing`, `completed`, `failed`.

---

### `GET /results/{job_id}`

Retrieve analysis results.

**Response:** `200 OK`
```json
{
  "job_id": "abc123",
  "disclaimer": "For reference only. Not medical advice. Clinician must use their judgment.",
  "total_slices": 64,
  "abnormal_slices": [
    {
      "slice_index": 47,
      "confidence": 0.82,
      "finding": "Suspicious region detected in slice 47 (confidence: 0.82).",
      "bbox": {"x": 120, "y": 95, "width": 45, "height": 38},
      "image_url": "/static/results/abc123/slice_47.png"
    },
    {
      "slice_index": 63,
      "confidence": 0.74,
      "finding": "Suspicious region detected in slice 63 (confidence: 0.74).",
      "bbox": {"x": 200, "y": 180, "width": 30, "height": 25},
      "image_url": "/static/results/abc123/slice_63.png"
    }
  ]
}
```

**Errors:**
- `404` — Job not found or results expired

---

## 9. Performance Expectations

| Metric | Target (CPU) | Target (GPU) |
|---|---|---|
| DICOM parse + validate (50 slices) | < 3 s | < 3 s |
| Inference (50 slices, batch) | < 30 s | < 5 s |
| Postprocess + render | < 5 s | < 3 s |
| **Total wall time (50 slices)** | **< 40 s** | **< 12 s** |

### Optimization Notes

- **Batch inference:** Process slices in batches (e.g., 8–16) rather than one-by-one.
- **Model caching:** Load model weights once at startup; keep in memory.
- **Image caching:** Cache rendered PNGs on disk in `TEMP_DIR`; serve directly via static files.
- **Lazy imports:** Import `torch`/heavy libs only when first needed to keep startup fast.

---

## 10. Testing Strategy

### Unit Tests

- `test_dicom_parser.py` — Validate DICOM reading, PHI tag detection, modality filtering, slice sorting, windowing.
- `test_inference.py` — Test with a tiny/mock model; verify output shape, score range [0,1], CAM dimensions.
- `test_postprocess.py` — Threshold logic, bounding box extraction, finding text generation.

### API Smoke Tests

- `test_routes.py`:
  - `POST /upload` with valid synthetic DICOM → 202
  - `POST /upload` with non-DICOM file → 400
  - `POST /upload` with oversized payload → 413
  - `GET /jobs/{id}` → returns valid status
  - `GET /results/{id}` → returns results with disclaimer field present

### End-to-End (Optional)

- `test_e2e.py`:
  - Upload a small synthetic series (5 slices).
  - Poll `/jobs/{id}` until `completed`.
  - Fetch `/results/{id}` and assert ≥0 abnormal slices returned.
  - Verify disclaimer string is present in response.

### Running Tests

```bash
pytest tests/ -v --tb=short
# With coverage:
pytest tests/ --cov=app --cov-report=term-missing
```

---

## 11. Task Breakdown — Next 5–10 Commits

| # | Commit Title | Delivers |
|---|---|---|
| 1 | `scaffold: project skeleton, pyproject.toml, gitignore` | Empty app structure, dependency list, .env.example, .gitignore with data/ and models/ |
| 2 | `feat: FastAPI app factory with health endpoint` | `app/main.py`, config loading, `GET /health` returns 200, uvicorn entrypoint |
| 3 | `feat: DICOM upload endpoint with validation` | `POST /upload`, file size check, DICOM magic byte validation, temp storage, job ID generation |
| 4 | `feat: DICOM parser service` | `dicom_parser.py` — read pixel data, sort slices, apply windowing, strip PHI tags |
| 5 | `feat: inference stub with mock model` | `inference.py` — loads a dummy model, returns random scores + synthetic CAMs, batch interface |
| 6 | `feat: postprocessing and ROI extraction` | `postprocess.py` — threshold CAM, extract bounding box, generate finding sentence |
| 7 | `feat: results endpoint and HTML results page` | `GET /results/{id}`, Jinja2 template with annotated images, disclaimer banner |
| 8 | `feat: upload UI and job polling page` | `index.html` upload form, `app.js` polls job status, redirects to results |
| 9 | `feat: real model integration` | Swap mock model for actual classifier (e.g., pretrained ResNet on chest CT), update `download_model.py` |
| 10 | `test: unit tests, smoke tests, CI config` | Full test suite, `pytest` in CI, coverage threshold |

---

## 12. How to Work with Claude Code

### Requesting Changes

- State what you want changed, not how to change it. Example: "Add a confidence threshold slider to the results page" rather than "edit results.html line 42."
- If a task is ambiguous, Claude Code will ask clarifying questions before writing code.
- Reference this file (`CLAUDE.md`) for conventions — Claude Code will follow them.

### Writing Tasks (Definition of Done)

A task is **done** when:
1. Code is written and follows the conventions in Section 5.
2. Relevant tests pass (`pytest tests/ -v`).
3. `ruff check . && ruff format --check .` passes with no errors.
4. The disclaimer is present wherever results are displayed.
5. No PHI is stored, logged, or committed.

### Rules for Claude Code

- **Do not invent medical claims.** Finding text must be template-based and factual (e.g., "Suspicious region detected" with a confidence score). Never generate diagnostic language like "malignant" or "cancer detected."
- **Always include the disclaimer** on any page or API response that shows results: _"For reference only. Not medical advice. Clinician must use their judgment."_
- **Prefer simple MVP solutions.** Don't over-engineer. If a dict works, don't add a database. If a template string works, don't add an LLM.
- **No hidden dependencies.** Every dependency must be in `pyproject.toml`. No `pip install` in code comments.
- **Ask before adding large dependencies** (e.g., TensorFlow, MONAI). Prefer PyTorch + torchvision for model inference.

---

## 13. Explicit "DO NOT" List

- **DO NOT** provide medical advice or diagnostic conclusions in any output.
- **DO NOT** commit, log, or display Protected Health Information (PHI).
- **DO NOT** claim or imply the system is clinically validated, FDA-cleared, or suitable for clinical use.
- **DO NOT** build complex distributed architecture (no Celery, no Redis, no Kubernetes) for this demo. Use in-process `asyncio` tasks or `BackgroundTasks`.
- **DO NOT** use an LLM to generate finding text — use deterministic templates only.
- **DO NOT** store results permanently — all data is ephemeral and auto-deleted.
- **DO NOT** expose the app to the public internet without authentication (out of scope for demo, but note it).
- **DO NOT** skip the disclaimer on any result-facing surface (API response, HTML page, exported file).
- **DO NOT** add GPU as a hard requirement — the app must start and run on CPU-only machines.
- **DO NOT** vendor or commit model weight files to the repository.
