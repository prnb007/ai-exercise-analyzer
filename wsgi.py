#!/usr/bin/env python3
"""
WSGI entry point for gunicorn
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from app_mongodb import app

# Initialize database and gamification when imported by gunicorn
try:
    from app_mongodb import initialize_database, ensure_daily_challenge
    initialize_database()
    ensure_daily_challenge()
    print("Database and gamification initialized successfully")
except Exception as e:
    print(f"Initialization error: {e}")

if __name__ == "__main__":
    app.run()
