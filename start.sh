#!/bin/bash

# Get port from environment variable, default to 5000
PORT=${PORT:-5000}

echo "Starting AI Exercise Analyzer on port $PORT"

# Start gunicorn with the correct port
exec gunicorn app_mongodb:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
