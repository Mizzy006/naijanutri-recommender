#!/bin/bash

# 1. Start the FastAPI backend in the background (runs on port 8000 inside the container)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# 2. Start the Streamlit frontend in the foreground (binds to Render's public port)
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0