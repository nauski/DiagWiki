#!/bin/bash

# DiagWiki Launch Script
# Starts both backend and frontend servers

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    ollama serve &
    sleep 2
fi

# Start frontend in background
echo "Starting frontend server..."
cd frontend
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 2

# Start backend in foreground (logs will be displayed)
echo "Starting backend server..."
echo ""
echo "DiagWiki is running!"
echo "   Backend:  http://localhost:8001"
echo "   Frontend: http://localhost:5173"
echo ""
echo "Backend logs:"
echo "----------------------------------------"

# Trap Ctrl+C and cleanup
trap "echo ''; echo 'Stopping servers...'; kill $FRONTEND_PID 2>/dev/null; exit" INT

cd backend
conda run -n diagwiki --no-capture-output python main.py

