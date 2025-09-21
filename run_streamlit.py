#!/usr/bin/env python3
"""
Streamlit Deployment Script
Simple script to run the Streamlit application with proper configuration
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    print("🚀 Starting Recommendation System Streamlit App...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'streamlit_app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true',
            '--browser.gatherUsageStats=false'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Streamlit app...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
