@echo off
REM quickstart.bat - sets up VisionGuard AI and launches the full stack on Windows.
REM Usage: double-click this file, or run it from cmd/PowerShell.

echo ==============================================
echo  VisionGuard AI - Quickstart
echo ==============================================

IF NOT EXIST venv (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) ELSE (
    echo [1/5] Virtual environment already exists, skipping.
)

call venv\Scripts\activate.bat

echo [2/5] Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo [3/5] Generating a small synthetic dataset for smoke-testing...
python src\detection\generate_synthetic_demo_data.py --out_dir data\processed\synthetic_demo --n_images 100

echo [4/5] Starting API on http://localhost:8000 ...
start "VisionGuard API" cmd /k "call venv\Scripts\activate.bat && uvicorn api.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > NUL

echo [5/5] Starting dashboard on http://localhost:8501 ...
echo.
echo Open your browser to: http://localhost:8501
echo API docs available at: http://localhost:8000/docs
echo.
echo Close this window and the API window to stop the services.

streamlit run dashboard\app.py
