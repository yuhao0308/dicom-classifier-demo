FROM python:3.11-slim

WORKDIR /app

# System deps for pydicom / Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends libjpeg62-turbo-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (no torch in production mock mode)
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "pydicom>=2.4" \
    "Pillow>=10.0" \
    "python-multipart>=0.0.6" \
    "jinja2>=3.1" \
    "pydantic-settings>=2.1" \
    "structlog>=24.1" \
    "numpy>=1.26" \
    "scipy>=1.12"

COPY app/ ./app/

# Create required directories
RUN mkdir -p /tmp/dicom-demo models data

ENV USE_MOCK_MODEL=true
ENV TEMP_DIR=/tmp/dicom-demo
ENV LOG_LEVEL=INFO
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
