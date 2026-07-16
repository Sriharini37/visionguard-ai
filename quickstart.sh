#!/usr/bin/env bash
# quickstart.sh — sets up VisionGuard AI and launches the full stack locally.
# Usage:
#   chmod +x quickstart.sh
#   ./quickstart.sh
set -e

echo "=============================================="
echo " VisionGuard AI — Quickstart"
echo "=============================================="

# 1. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment already exists, skipping."
fi

source venv/bin/activate

# 2. Install dependencies
echo "[2/5] Installing dependencies (this can take a few minutes the first time)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q || {
    echo "Some packages failed (likely tensorrt/pycuda — those need an NVIDIA Jetson/CUDA machine)."
    echo "Continuing anyway; the dashboard and API don't need them."
}

# 3. Generate synthetic demo data so the pipeline can be smoke-tested
echo "[3/5] Generating a small synthetic dataset for smoke-testing..."
python src/detection/generate_synthetic_demo_data.py --out_dir data/processed/synthetic_demo --n_images 100 || true

# 4. Start the FastAPI backend in the background
echo "[4/5] Starting API on http://localhost:8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID (logs: api.log)"

sleep 2

# 5. Start the Streamlit dashboard in the foreground
echo "[5/5] Starting dashboard on http://localhost:8501 ..."
echo ""
echo "Open your browser to: http://localhost:8501"
echo "API docs available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services."

trap "kill $API_PID 2>/dev/null" EXIT

streamlit run dashboard/app.py
