"""Launch the WorkSense AI Streamlit dashboard."""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"], check=True)
