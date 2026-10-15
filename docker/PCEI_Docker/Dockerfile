FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Non-interactive apt installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY pcei_framework /app/pcei_framework

# Copy execution scripts
COPY setup_data.py /app/
COPY run_training.sh /app/

# Make executable
RUN chmod +x /app/run_training.sh

# Default command
CMD ["/app/run_training.sh"]
