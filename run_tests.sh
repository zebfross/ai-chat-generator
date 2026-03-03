#!/bin/bash
# Start mock servers + bot, run integration tests, tear everything down.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Load base env (Pinecone creds, etc.) then override with mock URLs
set -a; source .env; set +a
export CHATWOOT_URL=http://localhost:5051
export ANTHROPIC_BASE_URL=http://localhost:5052
export PORT=5050

BOT_LOG=/tmp/ai-chat-bot-test.log

# Kill bot on exit
cleanup() {
    if [ -n "$BOT_PID" ]; then
        kill "$BOT_PID" 2>/dev/null
        wait "$BOT_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# Check for port conflicts
for p in 5050 5051 5052; do
    if lsof -ti :$p > /dev/null 2>&1; then
        echo "ERROR: Port $p is already in use (PID $(lsof -ti :$p))"
        exit 1
    fi
done

# Start the bot (output to log file to keep test output clean)
echo "==> Starting bot on :$PORT (loading models...)"
venv/bin/python app.py > "$BOT_LOG" 2>&1 &
BOT_PID=$!

# Wait for bot to accept connections
echo -n "==> Waiting for bot"
for i in $(seq 1 120); do
    if curl -sf http://localhost:$PORT/ > /dev/null 2>&1; then
        echo " ready!"
        break
    fi
    if ! kill -0 "$BOT_PID" 2>/dev/null; then
        echo " FAILED (bot crashed)"
        echo ""
        echo "==> Bot logs:"
        cat "$BOT_LOG"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Run tests (mock Chatwoot + mock Anthropic start inside test_server.py)
echo ""
venv/bin/python test_server.py
RESULT=$?

if [ $RESULT -ne 0 ]; then
    echo ""
    echo "==> Bot logs (last 50 lines):"
    tail -50 "$BOT_LOG"
fi

exit $RESULT
