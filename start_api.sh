#!/bin/bash

# Set API key for authentication
export API_KEY="95bf9dc602b49ca3983733bb4ce16c38830887789c105089230b0469225cc539"

# Start the FastAPI server in the background
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /home/ubuntu/llm-api/api_server.log 2>&1 &

# Save the PID to a file for easy stopping later
echo $! > /home/ubuntu/llm-api/api_server.pid

echo "API server started with PID: $!"
echo "Logs are being written to: /home/ubuntu/llm-api/api_server.log"