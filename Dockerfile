# ThoraVis inference API — CPU by default (portable, no NVIDIA container
# runtime required). For GPU inference, install a CUDA-enabled torch build
# instead (see README "Docker" section) and run with `docker run --gpus all`.
FROM python:3.11-slim

WORKDIR /app

# System libs OpenCV needs at import time (libGL, libglib) that slim doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# Install CPU-only torch/torchvision from PyTorch's own CPU index first.
# PyPI's default torch wheel now pulls the full NVIDIA CUDA runtime in as
# pip dependencies (cuda-toolkit, nvidia-cublas, nvidia-cudnn, ...) even
# with no GPU involved — that alone bloats this image past 10GB. The CPU
# index avoids it entirely; pip then sees torch/torchvision already
# satisfy requirements-api.txt's >= pins and won't touch them again.
# requirements-api.txt (not requirements.txt) — the API doesn't need
# jupyter/notebook/matplotlib, which is most of this image's non-torch weight.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/

# The trained checkpoint (~350MB) is not baked into the image — mount it at
# runtime with an absolute host path, e.g.:
#   docker run -p 8000:8000 -v /absolute/path/to/thoravis/models:/app/models thoravis
ENV THORAVIS_CHECKPOINT=/app/models/best_thoravis.pt
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
