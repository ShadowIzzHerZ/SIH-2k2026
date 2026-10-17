@echo off
title SIH 2026 Dead Reckoning 10Hz Backend Engine
echo ====================================================================
echo  Starting SIH 2026 Dead Reckoning 10Hz Backend Engine (Windows)
echo ====================================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH!
    echo Please install Python 3.10+ and check 'Add Python to PATH'.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [1/3] Setting up Python virtual environment...
    python -m venv venv
)

REM SECRET_KEY is required: create .env from the template on first run
if not exist ".env" (
    echo Creating .env from .env.example - replace SECRET_KEY before deploying.
    copy .env.example .env >nul
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo [2/3] Verifying and installing requirements...
pip install -r requirements.txt

REM Start the server
echo [3/3] Launching FastAPI 10Hz Dead Reckoning Backend...
echo ====================================================================
echo  Testing Page: http://localhost:8000/
echo  API Docs:     http://localhost:8000/docs
echo ====================================================================
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
