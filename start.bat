@echo off
REM SMS Spam Detection Application Startup Script for Windows

echo 🚀 Starting SMS Spam Detection Application
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies. Please check your Python environment.
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ⚠️  requirements.txt not found. Skipping dependency installation.
)

REM Start the Flask application
echo 🌐 Starting Flask application...
echo 📍 Application will be available at: http://localhost:5001
echo 🛑 Press Ctrl+C to stop the server
echo.

python app.py

pause
