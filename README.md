# DICOM Classifier Demo

A lightweight web application that accepts a DICOM CT series upload (.zip), runs a ResNet-18 classification model to flag suspicious slices, and returns annotated results with ROI bounding boxes and per-slice findings.

**This is a demo/prototype only.** Not a diagnostic medical device. Not FDA-cleared or clinically validated.

## Quick Start

```bash
# Create and activate a virtual environment (Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Download/create model weights
python scripts/download_model.py

# Start the server
uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000, upload a .zip archive of DICOM CT slices, and view the results.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Upload page |
| `POST` | `/upload` | Upload a DICOM .zip archive (returns 202) |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/results/{job_id}` | Fetch results as JSON |
| `GET` | `/results/{job_id}/view` | View results as HTML |
| `GET` | `/health` | Health check |

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/classifier.pt` | Path to model weights |
| `MAX_SLICES` | `100` | Maximum slices per upload |
| `USE_GPU` | `false` | Enable CUDA inference |
| `TEMP_DIR` | `/tmp/dicom-demo` | Temporary file directory |
| `TEMP_RETENTION_SECONDS` | `3600` | Auto-delete temp files after (seconds) |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload size |
| `INFERENCE_BATCH_SIZE` | `8` | Slices per inference batch |
| `LOG_LEVEL` | `INFO` | Logging level |

## Running Tests

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing

# Lint and format check
ruff check .
ruff format --check .
```

## Project Structure

```
app/
├── main.py              # App factory, lifespan, middleware
├── config.py            # Settings via pydantic-settings
├── routes/
│   ├── upload.py        # POST /upload, GET /
│   ├── jobs.py          # GET /jobs/{id}
│   └── results.py       # GET /results/{id}, GET /results/{id}/view
├── services/
│   ├── dicom_parser.py  # DICOM reading, PHI stripping, windowing
│   ├── inference.py     # ResNet-18 + GradCAM inference
│   ├── postprocess.py   # CAM thresholding, bbox extraction, overlays
│   └── storage.py       # Temp file and job metadata management
├── templates/           # Jinja2 HTML templates
└── static/              # CSS and JS
```

## Disclaimer

For reference only. Not medical advice. Clinician must use their judgment.
