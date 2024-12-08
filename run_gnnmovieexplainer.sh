#!/bin/bash

# Navigate to the backend and run the app
echo "Starting the backend..."
cd backend
python3 app.py &
BACKEND_PID=$!
echo "Backend started with PID $BACKEND_PID"

# Navigate to the frontend and start it with npm
echo "Starting the frontend..."
cd ../frontend
npm start &
FRONTEND_PID=$!
echo "Frontend started with PID $FRONTEND_PID"

# Function to clean up processes on exit
cleanup() {
  echo "Stopping the backend and frontend..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  echo "Processes stopped. Exiting..."
}

# Trap signals and clean up
trap cleanup EXIT

# Wait for processes to end (press Ctrl+C to terminate)
wait
