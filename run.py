import subprocess
import sys
import time
import os

# 1. Force the API URL to localhost since both apps will share the same container
os.environ["API_URL"] = "http://localhost:8000/api"

print("🚀 Starting FastAPI backend on port 8000...")
backend = subprocess.Popen([
    sys.executable, "-m", "uvicorn", "main:app", 
    "--host", "0.0.0.0", "--port", "8000"
])

# Give the backend 3 seconds to fully boot up before starting the frontend
time.sleep(3)

print("🎨 Starting Streamlit frontend...")
# Cloud providers like Render assign a dynamic port for external web traffic
web_port = os.environ.get("PORT", "8501") 
frontend = subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "app.py", 
    "--server.port", web_port, "--server.address", "0.0.0.0"
])

try:
    # Keep the script running
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("Shutting down services...")
    backend.terminate()
    frontend.terminate()
