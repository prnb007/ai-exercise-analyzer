#!/usr/bin/env python3
"""
WSGI entry point for gunicorn - Clean import approach
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')

# Import Flask and create a minimal app first
from flask import Flask

# Create a minimal Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Now import the main app and replace the minimal one
try:
    # Import the actual app from the module
    import app_mongodb
    
    # Replace our minimal app with the real one
    app = app_mongodb.app
    
    # Initialize database and gamification
    app_mongodb.initialize_database()
    app_mongodb.ensure_daily_challenge()
    print("Database and gamification initialized successfully")
    
except Exception as e:
    print(f"Error importing app: {e}")
    # Keep the minimal app as fallback
    pass

# This is the WSGI application that gunicorn will use
application = app

if __name__ == "__main__":
    app.run()
