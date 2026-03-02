# DICOM Classifier Demo

A lightweight web application that accepts a DICOM CT series upload, runs a classification model to flag suspicious slices, and returns annotated results with ROI bounding boxes and findings.

**This is a demo/prototype only.** Not a diagnostic medical device. Not FDA-cleared or clinically validated.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python scripts/download_model.py
uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000.

## Running Tests

```bash
pytest tests/ -v
```

## Disclaimer

For reference only. Not medical advice. Clinician must use their judgment.
