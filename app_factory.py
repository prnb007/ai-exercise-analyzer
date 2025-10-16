#!/usr/bin/env python3
"""
App factory for clean WSGI loading
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')

def create_app():
    """Create Flask application with proper initialization"""
    try:
        # Import the main app module
        from app_mongodb import app
        
        # Initialize database and gamification
        from app_mongodb import initialize_database, ensure_daily_challenge
        initialize_database()
        ensure_daily_challenge()
        print("Database and gamification initialized successfully")
        
        return app
        
    except Exception as e:
        print(f"Error creating app: {e}")
        # Create a minimal fallback app
        from flask import Flask
        app = Flask(__name__)
        app.secret_key = os.environ.get('SECRET_KEY', 'fallback-key')
        
        @app.route('/')
        def index():
            return "App initialization failed. Check logs."
            
        return app

# Create the app
app = create_app()

# This is the WSGI application that gunicorn will use
application = app

if __name__ == "__main__":
    app.run()
