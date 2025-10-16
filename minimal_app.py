#!/usr/bin/env python3
"""
Minimal Flask app for deployment - avoids problematic imports
"""
import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

# Set environment variables for production
os.environ.setdefault('FLASK_ENV', 'production')

# Create minimal Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Basic routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy", "message": "App is running"})

# This is the WSGI application that gunicorn will use
application = app

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
