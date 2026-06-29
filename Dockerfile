# Container for the PetVision REST API (CPU inference).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch/torchvision first (much smaller than the CUDA build),
# then the remaining dependencies.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
