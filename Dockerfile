FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    curl \
    build-essential \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Set the working directory
WORKDIR /app

# Create cache directory for Hugging Face models
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Install PyTorch with CUDA 12.8 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables for HF cache
ENV HF_HOME=/app/.cache
ENV DIFFUSERS_CACHE=/app/.cache

# Run your FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
