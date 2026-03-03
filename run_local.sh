#!/bin/bash
cd /Users/zeb/Documents/GitHub/ai-chat-generator
set -a
source .env
set +a
export PORT=5050
exec venv/bin/python app.py
