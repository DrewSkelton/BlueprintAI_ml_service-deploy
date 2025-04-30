FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    build-essential \
    curl \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Install PyTorch with CUDA 12.8 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy requirements file and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Set environment variables for Hugging Face cache
ENV HF_HOME=/app/.cache
ENV DIFFUSERS_CACHE=/app/.cache

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
