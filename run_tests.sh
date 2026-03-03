#!/bin/bash
# Start mock servers + bot, run integration tests, tear everything down.
# If the bot is already running on :5050, reuses it instead of starting a new one.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Load base env (Pinecone creds, etc.) then override with test env
set -a; source .env; source .env-test; set +a

BOT_LOG=/tmp/ai-chat-bot-test.log
BOT_PID=""
WE_STARTED_BOT=false

# Only kill the bot if we started it
cleanup() {
    if $WE_STARTED_BOT && [ -n "$BOT_PID" ]; then
        echo "==> Stopping bot (PID $BOT_PID)"
        kill "$BOT_PID" 2>/dev/null
        wait "$BOT_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# Check if bot is already running
if lsof -ti :$PORT > /dev/null 2>&1 > /dev/null 2>&1; then
    echo "==> Bot already running on :$PORT, reusing it"
else
    # Check for port conflicts on mock server ports only
    for p in 5051 5052; do
        if lsof -ti :$p > /dev/null 2>&1; then
            echo "ERROR: Port $p is already in use (PID $(lsof -ti :$p))"
            exit 1
        fi
    done

    # Start the bot (output to log file to keep test output clean)
    echo "==> Starting bot on :$PORT (loading models...)"
    venv/bin/python app.py > "$BOT_LOG" 2>&1 &
    BOT_PID=$!
    WE_STARTED_BOT=true

    # Wait for bot to accept connections
    echo -n "==> Waiting for bot"
    for i in $(seq 1 120); do
        if lsof -ti :$PORT > /dev/null 2>&1 > /dev/null 2>&1; then
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
fi

# Run tests (mock Chatwoot + mock Anthropic start inside test_server.py)
echo ""
venv/bin/python test_server.py
RESULT=$?

if [ $RESULT -ne 0 ] && $WE_STARTED_BOT; then
    echo ""
    echo "==> Bot logs (last 50 lines):"
    tail -50 "$BOT_LOG"
fi

exit $RESULT
