#!/usr/bin/env python3
"""
🚀 Pokemon LoRA Generator - Web App Launcher
Simple launcher for the Streamlit web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit web app"""
    print("🚀 Pokemon LoRA Generator - Web Interface")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found!")
    except ImportError:
        print("❌ Streamlit not installed!")
        print("💡 Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed!")
    
    # Get the path to streamlit_app.py
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print("❌ streamlit_app.py not found!")
        sys.exit(1)
    
    print("🌐 Starting web interface...")
    print("💡 The app will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("")
    print("⚠️ To stop the app, press Ctrl+C in this terminal")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web app stopped!")
    except Exception as e:
        print(f"❌ Error launching web app: {e}")
        print("💡 Try running manually: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()