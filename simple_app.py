#!/usr/bin/env python3
"""
Ultra-simple Flask app for deployment
"""
from flask import Flask, jsonify

# Create the simplest possible Flask app
app = Flask(__name__)
app.secret_key = 'simple-key-for-testing'

@app.route('/')
def hello():
    return jsonify({"message": "Hello World! App is working!", "status": "success"})

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

# WSGI application
application = app

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
