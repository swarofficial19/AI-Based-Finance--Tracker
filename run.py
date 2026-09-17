import subprocess
import sys
import time
import os

# 1. Use 127.0.0.1 explicitly to avoid IPv6 routing issues in cloud containers
os.environ["API_URL"] = "http://127.0.0.1:8000/api"

print("🚀 Starting FastAPI backend on port 8000...")
# Added stdout/stderr so if the backend crashes, the exact error prints to your Render logs!
backend = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

# 2. Give the cloud container 8 seconds to fully boot the FastAPI server
print("⏳ Waiting for backend to initialize...")
time.sleep(8)

print("🎨 Starting Streamlit frontend...")
web_port = os.environ.get("PORT", "8501") 
frontend = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", web_port, "--server.address", "0.0.0.0"],
    stdout=sys.stdout,
    stderr=sys.stderr
)

try:
    # Keep the script running
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    print("Shutting down services...")
    backend.terminate()
    frontend.terminate()
