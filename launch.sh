#!/bin/bash

# DiagWiki Launch Script
# Sets up environment and starts both backend and frontend servers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get local IP address for network access
get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost"
}

LOCAL_IP=$(get_local_ip)

# Cleanup function to kill all child processes
cleanup() {
    echo ''
    echo 'Stopping servers...'

    # Kill frontend (npm and its children)
    if [ ! -z "$FRONTEND_PID" ]; then
        pkill -P $FRONTEND_PID 2>/dev/null || true
        kill $FRONTEND_PID 2>/dev/null || true
    fi

    # Kill backend python process
    pkill -f "python3* main.py" 2>/dev/null || true

    # Kill any remaining vite or uvicorn processes
    pkill -f "vite" 2>/dev/null || true
    pkill -f "uvicorn" 2>/dev/null || true

    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

echo "========================================"
echo "  DiagWiki Setup & Launch"
echo "========================================"
echo ""

# Check if Claude Code CLI is available
echo "Checking prerequisites..."
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code CLI not found."
    echo "   Install it with: npm install -g @anthropic-ai/claude-code"
    echo "   Then authenticate: claude"
    exit 1
fi
echo "✅ Claude Code CLI found"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi
echo "✅ Python 3 found"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 20+"
    exit 1
fi
echo "✅ Node.js found"

echo ""

# Find available ports
find_available_port() {
    local START_PORT=$1
    local PORT=$START_PORT
    while lsof -ti:$PORT > /dev/null 2>&1; do
        PORT=$((PORT + 1))
        if [ $PORT -gt $((START_PORT + 100)) ]; then
            echo "❌ Could not find available port near $START_PORT"
            exit 1
        fi
    done
    echo $PORT
}

echo "Finding available ports..."
FRONTEND_PORT=$(find_available_port 5173)
BACKEND_PORT=$(find_available_port 8001)
echo "✅ Using frontend port: $FRONTEND_PORT"
echo "✅ Using backend port: $BACKEND_PORT"

echo ""

# Setup backend
echo "Setting up backend..."
cd "$SCRIPT_DIR/backend"

if [ ! -d "venv" ]; then
    echo "   Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "   Activating virtual environment..."
source venv/bin/activate

if [ -f "requirements.txt" ]; then
    echo "   Installing Python dependencies..."
    pip install -q -r requirements.txt
fi

echo "✅ Backend ready"

# Setup frontend
echo ""
echo "Setting up frontend..."
cd "$SCRIPT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "   Installing npm dependencies..."
    npm install --silent
fi

echo "✅ Frontend ready"

echo ""
echo "========================================"
echo "  Starting servers..."
echo "========================================"
echo ""

# Start frontend in background with network access
cd "$SCRIPT_DIR/frontend"
VITE_BACKEND_PORT=$BACKEND_PORT npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT > /dev/null 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

# Display access information
echo "DiagWiki is running!"
echo ""
echo "  Local access:"
echo "    Frontend: http://localhost:$FRONTEND_PORT"
echo "    Backend:  http://localhost:$BACKEND_PORT"
echo ""
echo "  Network access:"
echo "    Frontend: http://$LOCAL_IP:$FRONTEND_PORT"
echo "    Backend:  http://$LOCAL_IP:$BACKEND_PORT"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""
echo "========================================"
echo "  Backend logs:"
echo "========================================"

# Start backend in foreground (logs will be displayed)
cd "$SCRIPT_DIR/backend"
source venv/bin/activate
PORT=$BACKEND_PORT python3 main.py
