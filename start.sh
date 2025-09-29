#!/bin/bash

# SMS Spam Detection Application Startup Script

echo "🚀 Starting SMS Spam Detection Application"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check your Python environment."
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "⚠️  requirements.txt not found. Skipping dependency installation."
fi

# Start the Flask application
echo "🌐 Starting Flask application..."
echo "📍 Application will be available at: http://localhost:5001"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python3 app.py
