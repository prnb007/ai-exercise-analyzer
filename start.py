#!/usr/bin/env python3
"""
Production startup script for Railway deployment
"""
import os
import sys

def main():
    """Start the application with production settings"""
    # Set production environment
    os.environ.setdefault('FLASK_ENV', 'production')
    
    # Import and run the app
    from app_mongodb import app
    
    # Get port from Railway environment
    port = int(os.environ.get('PORT', 5003))
    
    print(f"Starting production server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()
