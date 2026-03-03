#!/usr/bin/env python3
"""
Mock Chatwoot + Mock Anthropic servers + integration tests for the webhook flow.

Both external dependencies (Chatwoot and Anthropic) are fully mocked, so tests
run without any real API calls.

Usage:
  1. Start the bot in another terminal:
     cd /Users/zeb/Documents/GitHub/ai-chat-generator
     source .env
     CHATWOOT_URL=http://localhost:5051 ANTHROPIC_BASE_URL=http://localhost:5052 PORT=5050 venv/bin/python app.py

  2. Run tests:
     venv/bin/python test_server.py
"""

import json
import re
import sys
import threading
import time
import uuid

import logging
import requests
from flask import Flask, jsonify, request as flask_request

# Suppress Flask/werkzeug request logs from the mock servers
logging.getLogger("werkzeug").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Mock Chatwoot server (port 5051)
# ---------------------------------------------------------------------------
mock_chatwoot = Flask("mock_chatwoot")

_cw_canned_history = {"payload": []}
_cw_captured_replies = []
_cw_lock = threading.Lock()


def set_chatwoot_history(payload_list):
    """Configure what GET /messages will return."""
    global _cw_canned_history
    with _cw_lock:
        _cw_canned_history = {"payload": payload_list}


def get_chatwoot_replies():
    with _cw_lock:
        return list(_cw_captured_replies)


def clear_chatwoot():
    global _cw_canned_history
    with _cw_lock:
        _cw_canned_history = {"payload": []}
        _cw_captured_replies.clear()


@mock_chatwoot.route(
    "/api/v1/accounts/<int:aid>/conversations/<int:cid>/messages",
    methods=["GET"],
)
def cw_get_messages(aid, cid):
    with _cw_lock:
        return jsonify(_cw_canned_history)


@mock_chatwoot.route(
    "/api/v1/accounts/<int:aid>/conversations/<int:cid>/messages",
    methods=["POST"],
)
def cw_post_message(aid, cid):
    data = flask_request.get_json(silent=True) or {}
    with _cw_lock:
        _cw_captured_replies.append(
            {
                "account_id": aid,
                "conversation_id": cid,
                "content": data.get("content", ""),
            }
        )
    return jsonify({"id": len(_cw_captured_replies), "content": data.get("content", "")})


# ---------------------------------------------------------------------------
# Mock Anthropic server (port 5052)
# ---------------------------------------------------------------------------
mock_anthropic = Flask("mock_anthropic")

_ant_responses = []       # canned responses, consumed FIFO
_ant_requests = []        # captured request bodies
_ant_lock = threading.Lock()


def _make_text_response(text):
    """Build a Messages API text response."""
    return {
        "id": f"msg_mock_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def _make_tool_use_response(tool_name, tool_input):
    """Build a Messages API tool_use response."""
    return {
        "id": f"msg_mock_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_mock_{uuid.uuid4().hex[:12]}",
                "name": tool_name,
                "input": tool_input,
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def set_anthropic_responses(responses):
    """Queue canned responses (consumed in order)."""
    with _ant_lock:
        _ant_responses.clear()
        _ant_responses.extend(responses)


def get_anthropic_requests():
    with _ant_lock:
        return list(_ant_requests)


def clear_anthropic():
    with _ant_lock:
        _ant_responses.clear()
        _ant_requests.clear()


@mock_anthropic.route("/v1/messages", methods=["POST"])
def ant_messages():
    data = flask_request.get_json(silent=True) or {}
    with _ant_lock:
        _ant_requests.append(data)
        if _ant_responses:
            resp = _ant_responses.pop(0)
        else:
            resp = _make_text_response("(mock fallback — no canned response queued)")
    return jsonify(resp)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
BOT_URL = "http://localhost:5050"
CW_PORT = 5051
ANT_PORT = 5052


def send_webhook(
    content,
    sender_name="John",
    sender_email="john@example.com",
    conversation_id=1,
    account_id=1,
):
    payload = {
        "event": "message_created",
        "message_type": "incoming",
        "content": content,
        "conversation": {"id": conversation_id},
        "account": {"id": account_id},
        "sender": {"name": sender_name, "email": sender_email},
    }
    return requests.post(f"{BOT_URL}/webhook", json=payload, timeout=120)


def _flatten_system(system_field):
    """Normalise the system field to a plain string.

    The Anthropic SDK may send system as a string or as a list of
    content blocks (e.g. [{"type": "text", "text": "..."}]).
    """
    if isinstance(system_field, str):
        return system_field
    if isinstance(system_field, list):
        return "\n".join(
            b["text"] for b in system_field if isinstance(b, dict) and "text" in b
        )
    return str(system_field or "")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_customer_info_in_prompt():
    """Verify customer name + email appear in the system prompt sent to Anthropic."""
    clear_chatwoot()
    clear_anthropic()
    set_chatwoot_history([])
    set_anthropic_responses([
        _make_text_response("Hi John! How can I help you today?"),
    ])

    resp = send_webhook("Hi, I have a question about tradelines.")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    reqs = get_anthropic_requests()
    assert len(reqs) == 1, f"Expected 1 Anthropic request, got {len(reqs)}"

    system = _flatten_system(reqs[0].get("system", ""))

    assert "John" in system, (
        f"Customer name 'John' not found in system prompt:\n{system[:300]}"
    )
    assert "john@example.com" in system, (
        f"Customer email not found in system prompt:\n{system[:300]}"
    )
    # Pinecone context header (always present even if Pinecone returns 0 results)
    assert "similar" in system.lower() or "past" in system.lower(), (
        f"Pinecone context header missing from system prompt:\n{system[:300]}"
    )

    # Reply should have been forwarded to Chatwoot
    replies = get_chatwoot_replies()
    assert len(replies) >= 1, "No reply sent to Chatwoot"
    assert "john" in replies[0]["content"].lower()

    print(f"  System prompt has customer name: OK")
    print(f"  System prompt has customer email: OK")
    print(f"  System prompt has Pinecone context: OK")
    print(f"  Reply forwarded to Chatwoot: {replies[0]['content'][:100]}")


def test_conversation_history_forwarded():
    """Verify prior Chatwoot messages are forwarded to Anthropic as message history."""
    clear_chatwoot()
    clear_anthropic()
    set_chatwoot_history([
        {"id": 1, "content": "Hi, how do tradelines work?", "message_type": 0},
        {
            "id": 2,
            "content": "Great question! Tradelines are credit accounts on your report.",
            "message_type": 1,
        },
    ])
    set_anthropic_responses([
        _make_text_response("Our most affordable option starts at $200."),
    ])

    resp = send_webhook("What's the cheapest option you have?")
    assert resp.status_code == 200

    reqs = get_anthropic_requests()
    assert len(reqs) == 1

    messages = reqs[0].get("messages", [])

    # Should have prior history (user + assistant) plus the new user message
    assert len(messages) >= 3, (
        f"Expected >= 3 messages (history + new), got {len(messages)}: "
        f"{json.dumps(messages, indent=2)[:400]}"
    )

    # First message should be from user (the history)
    assert messages[0]["role"] == "user", f"First message role: {messages[0]['role']}"
    assert "tradeline" in messages[0]["content"].lower()

    # Second should be assistant (from history)
    assert messages[1]["role"] == "assistant", f"Second message role: {messages[1]['role']}"

    # Last user message should be the new one
    last_user_msgs = [m for m in messages if m["role"] == "user"]
    assert any("cheapest" in m["content"].lower() for m in last_user_msgs), (
        f"New user message not found. User messages: "
        f"{[m['content'][:80] for m in last_user_msgs]}"
    )

    print(f"  Messages sent to Anthropic: {len(messages)}")
    print(f"  Roles: {[m['role'] for m in messages]}")
    print(f"  History user msg: {messages[0]['content'][:80]}")
    print(f"  History assistant msg: {messages[1]['content'][:80]}")


def test_tool_use_flow():
    """Verify tool definitions are sent and tool_result round-trips correctly."""
    clear_chatwoot()
    clear_anthropic()
    set_chatwoot_history([
        {"id": 1, "content": "Hi, I need help with my account", "message_type": 0},
        {"id": 2, "content": "Of course! How can I help?", "message_type": 1},
    ])
    set_anthropic_responses([
        # 1st response: Claude decides to call the tool
        _make_tool_use_response("reset_password", {"email": "test@example.com"}),
        # 2nd response: Claude's final answer after seeing the tool result
        _make_text_response(
            "I've sent a password reset email to test@example.com. Please check your inbox!"
        ),
    ])

    resp = send_webhook("I need to reset my password, my email is test@example.com")
    assert resp.status_code == 200

    reqs = get_anthropic_requests()
    assert len(reqs) == 2, f"Expected 2 Anthropic requests (tool_use + final), got {len(reqs)}"

    # --- First request: should include tool definitions ---
    tools = reqs[0].get("tools", [])
    tool_names = [t["name"] for t in tools]
    assert "reset_password" in tool_names, f"reset_password not in tools: {tool_names}"
    assert "search_tradelines" in tool_names, f"search_tradelines not in tools: {tool_names}"
    assert "order_lookup" in tool_names, f"order_lookup not in tools: {tool_names}"

    # --- Second request: should contain the tool_result ---
    second_msgs = reqs[1].get("messages", [])
    has_tool_result = False
    for msg in second_msgs:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    has_tool_result = True
                    # The tool_result content should mention the reset attempt
                    print(f"  Tool result: {str(block.get('content', ''))[:120]}")
                    break
    assert has_tool_result, (
        f"tool_result not found in second request messages:\n"
        f"{json.dumps(second_msgs, default=str)[:500]}"
    )

    # Reply to Chatwoot should mention password/reset
    replies = get_chatwoot_replies()
    assert len(replies) >= 1, "No reply sent to Chatwoot"
    reply_lower = replies[0]["content"].lower()
    assert "password" in reply_lower or "reset" in reply_lower, (
        f"Reply doesn't mention password/reset: {replies[0]['content'][:200]}"
    )

    print(f"  Tools sent: {tool_names}")
    print(f"  Tool result round-tripped: OK")
    print(f"  Reply: {replies[0]['content'][:100]}")


def test_empty_message_no_api_call():
    """Empty content — webhook returns early, no Anthropic or Chatwoot calls."""
    clear_chatwoot()
    clear_anthropic()
    set_chatwoot_history([])

    resp = send_webhook("")
    assert resp.status_code == 200

    data = resp.json()
    assert data.get("reason") == "empty message", f"Expected 'empty message', got: {data}"

    # No Anthropic request should have been made
    reqs = get_anthropic_requests()
    assert len(reqs) == 0, f"Expected 0 Anthropic requests, got {len(reqs)}"

    # No reply to Chatwoot
    replies = get_chatwoot_replies()
    assert len(replies) == 0, f"Expected 0 Chatwoot replies, got {len(replies)}"

    print(f"  Webhook response: {data}")
    print(f"  Anthropic calls: 0 — OK")
    print(f"  Chatwoot replies: 0 — OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_flask(app, port):
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def main():
    # Start mock servers
    print(f"Starting mock Chatwoot on :{CW_PORT} and mock Anthropic on :{ANT_PORT}...")
    for app, port in [(mock_chatwoot, CW_PORT), (mock_anthropic, ANT_PORT)]:
        t = threading.Thread(target=_run_flask, args=(app, port), daemon=True)
        t.start()
    time.sleep(0.5)

    # Verify bot is reachable
    try:
        requests.get(f"{BOT_URL}/", timeout=5)
    except requests.ConnectionError:
        print(f"\nERROR: Cannot reach the bot at {BOT_URL}")
        print(
            f"Start it first:\n"
            f"  cd /Users/zeb/Documents/GitHub/ai-chat-generator\n"
            f"  source .env\n"
            f"  CHATWOOT_URL=http://localhost:{CW_PORT} "
            f"ANTHROPIC_BASE_URL=http://localhost:{ANT_PORT} "
            f"PORT=5050 venv/bin/python app.py"
        )
        sys.exit(1)

    tests = [
        ("Customer info in system prompt", test_customer_info_in_prompt),
        ("Conversation history forwarded", test_conversation_history_forwarded),
        ("Tool-use round-trip", test_tool_use_flow),
        ("Empty message — no API calls", test_empty_message_no_api_call),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print("  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
