#!/bin/bash
# Quick start script for multi-user chat server

echo "Starting Multi-User MCP Chat Server..."
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed!"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  or visit: https://github.com/astral-sh/uv"
    echo ""
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "WARNING: No virtual environment found. Creating one with uv..."
    uv venv
    source .venv/bin/activate
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, websockets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages from requirements.txt using uv pip..."
    uv pip install -r requirements.txt
fi

# Start the server
echo " Starting server on http://localhost:8000"
echo "Open multiple browser tabs to http://localhost:8000 to test multi-user chat"
echo "Or run: python3 multi_user_chat_client.py --username <name> in multiple terminals"
echo ""

# Use the venv's Python to ensure correct environment
if [ -f ".venv/bin/python3" ]; then
    .venv/bin/python3 multi_user_chat.py
elif [ -f "venv/bin/python3" ]; then
    venv/bin/python3 multi_user_chat.py
else
    python3 multi_user_chat.py
fi

