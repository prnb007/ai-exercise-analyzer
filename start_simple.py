#!/usr/bin/env python3
"""
Simple startup script for Railway deployment
"""
import os
import sys

# Set environment variables
os.environ.setdefault('FLASK_ENV', 'production')

# Import and run the simple app
from simple_app import app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
