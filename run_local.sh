#!/bin/bash
cd /Users/zeb/code/ai-chat-generator
set -a
source .env
source .env-test
set +a
export PORT=5050
exec venv/bin/python app.py
