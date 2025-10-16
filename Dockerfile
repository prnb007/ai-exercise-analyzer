# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV OPENCV_HEADLESS=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads temp

# Expose port
EXPOSE $PORT

# Create startup script
RUN echo '#!/bin/sh\nPORT=${PORT:-5000}\necho "Starting server on port $PORT"\ngunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app_mongodb:app' > /app/start.sh && chmod +x /app/start.sh

# Start command
CMD ["/app/start.sh"]
