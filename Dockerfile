# syntax=docker/dockerfile:1

# --- frontend build -------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- runtime --------------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app

# LightGBM needs libgomp at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.9" \
      "lightgbm>=4.7" "scikit-learn>=1.5" "pandas>=2.2" "numpy>=2.0" \
      "scipy>=1.14" "joblib>=1.4"

COPY src/ ./src/
COPY data/reference/ ./data/reference/
COPY results/eda/figures/ ./results/eda/figures/
# Artifacts are built on the host with `python src/serve/artifacts.py`, which
# gates on reproducing the documented cross-validation metric. Building them
# inside the image would need the training data, which is not redistributed.
COPY artifacts/ ./artifacts/
COPY --from=web /build/dist/ ./frontend/dist/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
