#!/bin/bash

# Check if PID file exists
if [ -f "/home/ubuntu/llm-api/api_server.pid" ]; then
  PID=$(cat /home/ubuntu/llm-api/api_server.pid)
  
  # Check if the process is still running
  if ps -p $PID > /dev/null; then
    echo "Stopping API server (PID: $PID)..."
    kill $PID
    echo "API server stopped."
  else
    echo "API server is not running (PID: $PID)."
  fi
  
  # Remove the PID file
  rm /home/ubuntu/llm-api/api_server.pid
else
  # Try to find the process if PID file doesn't exist
  PID=$(ps -ef | grep "uvicorn app.main:app" | grep -v grep | awk '{print $2}')
  
  if [ -n "$PID" ]; then
    echo "Stopping API server (PID: $PID)..."
    kill $PID
    echo "API server stopped."
  else
    echo "API server is not running."
  fi
fi