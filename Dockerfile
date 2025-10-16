# Multi-stage build to reduce image size
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Install PyTorch CPU version first
RUN pip install --no-cache-dir --user torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENCV_HEADLESS=1
ENV PATH="/root/.local/bin:$PATH"

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY app_mongodb.py .
COPY mongodb_config.py .
COPY mongodb_models.py .
COPY gamification_manager.py .
COPY forms.py .
COPY static/ static/
COPY templates/ templates/
COPY models/ models/

# Create necessary directories
RUN mkdir -p uploads temp

# Expose port (Railway will set this)
EXPOSE 5000

# Start command with proper port handling
CMD ["sh", "-c", "gunicorn app_mongodb:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120"]
