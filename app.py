import json
import logging
import os
import re
import sys
import time
import uuid
from urllib.parse import parse_qs

from dotenv import load_dotenv
load_dotenv()

start_time = time.time()
from datetime import datetime
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

try:
    import yaml  # pip install pyyaml
except ImportError:
    yaml = None
MyApp = Flask(__name__)

# Optional: set up root logger if you haven't already (Heroku reads stdout)
logging.basicConfig(level=logging.INFO)

CORS(
    MyApp,
    resources={r"/v1/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "X-API-Key"],
    methods=["POST", "GET", "OPTIONS"],
    max_age=86400,
)

SENSITIVE_HEADER_KEYS = {"authorization", "x-api-key", "proxy-authorization"}

from typing import Any, Dict, Tuple

SENSITIVE_HEADER_KEYS = {"authorization", "x-api-key", "proxy-authorization"}


def extract_message(req):
    """
    Returns (message_or_None, source, debug)
    source ∈ {"json","arguments_json","form","query","raw_kv",None}
    """
    debug = {
        "notes": [],
        "json_keys": None,
        "form_keys": None,
        "query_keys": list(req.args.keys()),
    }

    # --- RAW (cached) & content-type ---
    raw = req.get_data(cache=True, as_text=True) or ""
    ct = (req.headers.get("Content-Type") or "").lower()

    # --- JSON body (even if CT is wrong) ---
    data = req.get_json(silent=True)
    if isinstance(data, str):
        try:
            data = json.loads(data)
            debug["notes"].append("top_level_string_json_unwrapped")
        except Exception:
            data = None

    if isinstance(data, dict):
        debug["json_keys"] = list(data.keys())
        # direct message
        m = data.get("message")
        if isinstance(m, str) and m.strip():
            return m.strip(), "json", debug

        # arguments object or stringified
        args = data.get("arguments")
        if isinstance(args, dict):
            m = args.get("message")
            if isinstance(m, str) and m.strip():
                debug["notes"].append("arguments_object")
                return m.strip(), "arguments_json", debug
        elif isinstance(args, str):
            try:
                obj = json.loads(args)
                if isinstance(obj, dict):
                    m = obj.get("message")
                    if isinstance(m, str) and m.strip():
                        debug["notes"].append("arguments_string_json")
                        return m.strip(), "arguments_json", debug
            except Exception:
                pass
        # common aliases in JSON
        for k in ("input", "query", "q", "text", "prompt"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                debug["notes"].append(f"json_alias:{k}")
                return v.strip(), "json", debug

    # --- form (x-www-form-urlencoded or multipart) ---
    form = req.form.to_dict(flat=True) if getattr(req, "form", None) else {}
    debug["form_keys"] = list(form.keys())
    for k in ("message", "input", "query", "q", "text", "prompt", "arguments"):
        v = form.get(k)
        if v and isinstance(v, str) and v.strip():
            # if arguments looks like JSON, unwrap message
            if k == "arguments":
                try:
                    obj = json.loads(v)
                    if isinstance(obj, dict) and isinstance(obj.get("message"), str):
                        return obj["message"].strip(), "form", debug
                except Exception:
                    pass
            return v.strip(), "form", debug

    # --- query string (ALWAYS check this) ---
    for k in ("message", "input", "query", "q", "text", "prompt", "arguments"):
        v = req.args.get(k)
        if v and isinstance(v, str) and v.strip():
            # unwrap arguments if JSON-looking
            if k == "arguments":
                try:
                    obj = json.loads(v)
                    if isinstance(obj, dict) and isinstance(obj.get("message"), str):
                        return obj["message"].strip(), "query", debug
                except Exception:
                    pass
            return v.strip(), "query", debug

    # --- raw fallback like 'message=abc' ---
    if raw and "=" in raw and "&" not in raw and "{" not in raw:
        try:
            kv = parse_qs(raw, keep_blank_values=True)
            vlist = kv.get("message") or kv.get("arguments")
            if vlist and vlist[0].strip():
                return vlist[0].strip(), "raw_kv", debug
        except Exception:
            pass

    return None, None, debug


def _trunc(s: Any, n: int = 1000) -> Any:
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + "…"


def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    red = {}
    for k, v in headers.items():
        if k.lower() in SENSITIVE_HEADER_KEYS:
            red[k] = "<redacted>"
        else:
            red[k] = v
    return red


def _inspect_request(req) -> Tuple[Dict[str, Any], str, Any, Dict[str, Any]]:
    """
    Returns (summary_dict, raw_text, parsed_json, form_dict)
    - raw_text: full request body as text (safe cached read)
    - parsed_json: result of request.get_json(silent=True), with a second pass if it's a string
    - form_dict: flat form fields
    """
    # Safe cached read; Flask will reuse cached data later
    raw_bytes = req.get_data(cache=True) or b""
    try:
        raw_text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        raw_text = ""

    parsed_json = req.get_json(silent=True)
    if isinstance(parsed_json, str):
        # Sometimes body is a JSON-encoded string
        try:
            parsed_json = json.loads(parsed_json)
        except Exception:
            pass

    form = req.form.to_dict(flat=True) if getattr(req, "form", None) else {}

    summary = {
        "content_type": req.headers.get("Content-Type"),
        "content_length": req.headers.get("Content-Length"),
        "headers": _redact_headers(dict(req.headers)),
        "args": req.args.to_dict(flat=True),
        "form_keys": list(form.keys()),
        "file_keys": list(getattr(req, "files", {}).keys()),
        "raw_len": len(raw_text),
        "raw_preview": _trunc(raw_text, 2000),
        "json_type": type(parsed_json).__name__ if parsed_json is not None else None,
        "json_keys": (
            list(parsed_json.keys()) if isinstance(parsed_json, dict) else None
        ),
    }
    return summary, raw_text, parsed_json, form


def _log(event: str, **payload):
    # Compact JSON for easy grepping in Heroku logs
    try:
        logging.info(json.dumps({"event": event, **payload}, default=str))
    except Exception as e:
        logging.info(json.dumps({"event": event, "log_error": str(e)}))


import requests as http_requests
from anthropic import Anthropic
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


def timer(event_name):
    global start_time
    current_time = time.time()
    # print(event_name + f" {current_time - start_time:.2f}")
    start_time = current_time


timer("import transformer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_FILENAME = "api.yml"  # your file
SPEC_PATH = os.path.join(BASE_DIR, SPEC_FILENAME)

# Initialize Anthropic and Pinecone
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
pc = Pinecone(api_key=os.environ["pinecone_key"])
timer("create pinecone db")
# Define the Pinecone index name
db_index_name = "chat-history"

# Ensure the index exists or create it if not
# if db_index_name not in pc.list_indexes():
#    pc.create_index(
#        name=db_index_name,
#        dimension=384,  # Adjust to match the embedding model dimensionality
#        metric="cosine",
#        spec=ServerlessSpec(
#                cloud='aws',
#                region='us-east-1'
#            )
#    )
index = pc.Index(db_index_name)


# Load a pre-trained sentence-transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Compact and efficient
timer("load model")


def get_embedding(text):
    return model.encode(text).tolist()


# Function to store chat history in Pinecone
def store_chat(index, chat_message, response, timestamp):
    embedding = get_embedding(chat_message)
    metadata = {
        "chat_message": chat_message,
        "response": response,
        "timestamp": timestamp,
    }
    index.upsert([(str(uuid.uuid4()), embedding, metadata)])


# Function to search similar chats in Pinecone
def search_similar_chats(index, query, top_k=5):
    embedding = get_embedding(query)
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results


# Load chat history from a local JSON file
def load_chat_history(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return json.load(file)


@MyApp.route("/chat")
def hello():
    # Load chat history from file
    # chat_history_file = "sanitized_output.json"  # Replace with your file path
    # chat_history = load_chat_history(chat_history_file)

    # Store chat history in Pinecone
    # for entry in chat_history:
    #    chat_message = entry.get("chat_message")
    #    response = entry.get("response")
    #    timestamp = entry.get("timestamp", "unknown")
    #    try:
    #        store_chat(index, chat_message, response, timestamp)
    #    except:
    #        print("Error indexing message at ", timestamp)
    # print("Chat history successfully stored in Pinecone.")

    # Example: Search for similar chats
    args = request.args
    query = args.get("message")
    if not query:
        return "message parameter is required", 400
    customer_name = args.get("name")
    customer_email = args.get("email")

    results = search_similar_chats(index, query)
    timer("find similar chats")
    history = """You are a support chat agent for TradelineWorks.com. Be friendly, concise, and helpful.

Rules:
- If you don't know the answer, say you'll connect them with a specialist.
- Never make up pricing, guarantees, or specific timelines.
- Match the tone and style of the example responses below.

"""

    if customer_name or customer_email:
        history += "Current customer info:\n"
        if customer_name:
            history += f"- Name: {customer_name}\n"
        if customer_email:
            history += f"- Email: {customer_email}\n"
        history += "\n"

    history += "Similar past conversations for reference:\n"

    for result in results["matches"]:
        history += "\nCustomer: " + result["metadata"]["chat_message"]
        history += "\nAgent: " + result["metadata"]["response"]
        history += "\n---"

    logging.info("=== ANTHROPIC REQUEST ===")
    logging.info("Matches: %d", len(results["matches"]))
    logging.info("System: %s", history)
    logging.info("User: %s", query)

    reply, _ = _run_with_tools(history, [{"role": "user", "content": query}],
                              customer_email=customer_email)

    logging.info("=== ANTHROPIC RESPONSE ===")
    logging.info("Reply: %s", reply)

    timer("anthropic response")
    return reply


@MyApp.get("/openapi.yaml")
def serve_openapi_yaml():
    if not os.path.exists(SPEC_PATH):
        return Response(
            "Spec not found (api.yml)", status=404, mimetype="text/plain; charset=utf-8"
        )

    with open(SPEC_PATH, "rb") as f:
        data = f.read()

    resp = Response(data, status=200, mimetype="text/yaml; charset=utf-8")
    # Optional: allow third-party fetches
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


MyApp.url_map.strict_slashes = False


@MyApp.get("/v1/assist/suggest")
def assist_suggest():

    req_id = str(uuid.uuid4())
    t0 = time.time()

    # Grab API key but DO NOT log its value
    api_key = request.headers.get("X-API-Key")
    authed = bool(api_key and api_key == os.environ.get("api_key"))

    # Inspect EVERYTHING up front (doesn't consume the stream)
    req_summary, raw_text, parsed_json, form_dict = _inspect_request(request)

    # Extract message using your tolerant parser (recommended)
    message, source, parse_dbg = extract_message(
        request
    )  # returns (message|None, debug_dict)

    # Log a rich event before any early returns
    _log(
        "assist_suggest.request",
        req_id=req_id,
        method=request.method,
        path=request.path,
        endpoint=(request.url_rule.endpoint if request.url_rule else None),
        remote_addr=request.headers.get("X-Forwarded-For") or request.remote_addr,
        user_agent=request.headers.get("User-Agent"),
        has_api_key=bool(api_key),
        authed=authed,  # shows True/False, not the key
        route_matched=str(request.url_rule) if request.url_rule else None,
        **req_summary,  # content_type, content_length, headers (redacted), args, form_keys, files, raw_len/preview, json_keys
        message_preview=_trunc(message or "", 400),
        message_source=source,
        parse_debug=parse_dbg,  # which shape we detected, notes, keys, etc.
    )

    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != os.environ["api_key"]:
        return jsonify({"error": "unauthorized", "message": "Invalid API key"}), 401

    if not message:
        return (
            jsonify({"error": "validation_error", "message": "message is required"}),
            400,
        )

    results = search_similar_chats(index, message)
    timer("find similar chats")

    items: List[Dict[str, Any]] = []
    for m in results["matches"]:
        md = m.get("metadata") or {}
        # Reserve known keys; pass the rest through under metadata
        msg = md.get("chat_message") or md.get("message") or ""
        resp = md.get("response") or ""
        timestamp = md.get("timestamp")
        # normalize timestamp if it looks like a number
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.utcfromtimestamp(timestamp).isoformat() + "Z"

        # drop reserved keys from metadata passthrough
        passthrough = {
            k: v
            for k, v in md.items()
            if k not in {"chat_message", "message", "response", "timestamp"}
        }

        items.append(
            {
                "question": msg,
                "response": resp,
            }
        )

    sample_items = items[:3]  # avoid blowing up logs

    # Truncate large fields in the sample to keep logs readable
    def _truncate(s, n=200):
        return s if not isinstance(s, str) or len(s) <= n else s[:n] + "…"

    _log(
        "assist_suggest.response",
        req_id=req_id,
        total_items=len(items),
        sample=[
            {
                **{k: v for k, v in it.items() if k not in {"message", "response"}},
                "message_preview": _truncate(it.get("message", "")),
                "response_preview": _truncate(it.get("response", "")),
            }
            for it in sample_items
        ],
        elapsed_ms=int((time.time() - t0) * 1000),
    )

    return {
        "results": items
    }, 200


CHATWOOT_URL = os.environ.get("CHATWOOT_URL", "https://chat.inceptify.com")
CHATWOOT_BOT_TOKEN = os.environ.get("CHATWOOT_BOT_TOKEN", "")
CHATWOOT_USER_TOKEN = os.environ.get("CHATWOOT_USER_TOKEN", CHATWOOT_BOT_TOKEN)

SYSTEM_PROMPT = (
    "You are a TradelineWorks.com support chat agent. Keep responses short and concise. "
    "I'll give you previous chat requests we've received and how we responded to help you. "
    "Try to stay with the provided style and content.\n\n"
    "If you cannot help the customer, or they explicitly ask for a human agent, "
    "use the transfer_to_agent tool to hand the conversation to a live agent."
)

# ---------------------------------------------------------------------------
# Tool definitions — sent with every Anthropic request so Claude knows
# what actions it can take.  Add new tools here.
# ---------------------------------------------------------------------------
TW_BASE_URL = os.environ.get("TW_BASE_URL", "").rstrip("/")
TW_BOT_API_KEY = os.environ.get("TW_BOT_API_KEY", "")
TW_VERIFY_SSL = not any(h in TW_BASE_URL for h in ("localhost", ".local", "127.0.0.1"))

TOOLS = [
    {
        "name": "search_tradelines",
        "description": (
            "Search available tradelines (authorized-user credit lines) for sale. "
            "Use this when a customer asks about available tradelines, pricing, "
            "or wants to find a tradeline matching certain criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age_min": {
                    "type": "number",
                    "description": "Minimum age of the tradeline in years.",
                },
                "age_max": {
                    "type": "number",
                    "description": "Maximum age of the tradeline in years.",
                },
                "limit_min": {
                    "type": "number",
                    "description": "Minimum credit limit in dollars.",
                },
                "limit_max": {
                    "type": "number",
                    "description": "Maximum credit limit in dollars.",
                },
                "price_max": {
                    "type": "number",
                    "description": "Maximum price in dollars.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "order_lookup",
        "description": (
            "Look up a customer's active tradeline orders by their email address. "
            "Use this when a customer asks about the status of their order(s)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The customer's email address.",
                },
            },
            "required": ["email"],
        },
    },
    {
        "name": "reset_password",
        "description": (
            "Send a password-reset email to a TradelineWorks customer. "
            "Use this when a customer asks to reset or change their password."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The customer's email address.",
                },
            },
            "required": ["email"],
        },
    },
    {
        "name": "transfer_to_agent",
        "description": (
            "Transfer the conversation to a live human agent. "
            "Use this when you cannot help the customer or they ask for a human."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _tw_headers():
    return {"X-Bot-Api-Key": TW_BOT_API_KEY}


def _execute_tool(name: str, tool_input: dict, customer_email: str = None,
                   account_id: int = None, conversation_id: int = None) -> str:
    """Dispatch a tool call and return a plain-text result string."""
    if name == "search_tradelines":
        return _tool_search_tradelines(tool_input)
    if name == "order_lookup":
        return _tool_order_lookup(tool_input, customer_email)
    if name == "reset_password":
        return _tool_reset_password(tool_input, customer_email)
    if name == "transfer_to_agent":
        return _tool_transfer_to_agent()
    return f"Unknown tool: {name}"


def _tool_search_tradelines(tool_input: dict) -> str:
    """Search available tradelines via the TW API."""
    params = {}
    for key in ("age_min", "age_max", "limit_min", "limit_max", "price_max"):
        val = tool_input.get(key)
        if val is not None:
            params[key] = val

    try:
        resp = http_requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/cards",
            params=params,
            headers=_tw_headers(),
            timeout=15,
            verify=TW_VERIFY_SSL,
        )
        resp.raise_for_status()
        cards = resp.json()
    except Exception as e:
        logging.exception("search_tradelines error")
        return f"Error searching tradelines: {e}"

    if not cards:
        return "No tradelines found matching those criteria."

    lines = [f"Found {len(cards)} tradeline(s):\n"]
    for c in cards:
        lines.append(
            f"- {c['name']} | Age: {c['age_months']} months | "
            f"Limit: ${int(float(c['limit'])):,} | Price: ${c['price']} | "
            f"Stock: {c['stock_remaining']} | URL: {c['url']}"
        )
    return "\n".join(lines)


def _tool_order_lookup(tool_input: dict, customer_email: str = None) -> str:
    """Look up a customer's orders via the TW API."""
    email = tool_input.get("email", "").strip()
    if not email:
        return "Error: no email address provided."
    if not customer_email:
        return "Error: cannot look up orders — customer email is not verified for this session."
    if email.lower() != customer_email.lower():
        return f"Error: for security, I can only look up orders for the current customer ({customer_email})."

    try:
        resp = http_requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/orders",
            params={"email": email},
            headers=_tw_headers(),
            timeout=15,
            verify=TW_VERIFY_SSL,
        )
        if resp.status_code == 404:
            return f"No account found for {email}."
        resp.raise_for_status()
        orders = resp.json()
    except Exception as e:
        logging.exception("order_lookup error")
        return f"Error looking up orders: {e}"

    if not orders:
        return f"No active orders found for {email}."

    lines = [f"Found {len(orders)} order(s) for {email}:\n"]
    for o in orders:
        lines.append(
            f"- Order #{o['order_id']}: {o['card_name']} | "
            f"Status: {o['status']} | Started: {o['start_date'] or 'N/A'}\n"
            f"  Next step: {o['next_step']}"
        )
    return "\n".join(lines)


def _tool_reset_password(tool_input: dict, customer_email: str = None) -> str:
    """Send a password-reset email via the TW API."""
    email = tool_input.get("email", "").strip()
    if not email:
        return "Error: no email address provided."
    if not customer_email:
        return "Error: cannot reset password — customer email is not verified for this session."
    if email.lower() != customer_email.lower():
        return f"Error: for security, I can only reset the password for the current customer ({customer_email})."

    try:
        resp = http_requests.post(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/reset-password",
            json={"email": email},
            headers=_tw_headers(),
            timeout=15,
            verify=TW_VERIFY_SSL,
        )
        if resp.status_code == 404:
            return f"No account found for {email}."
        resp.raise_for_status()
    except Exception as e:
        logging.exception("reset_password error")
        return f"Error sending password reset: {e}"

    return f"A password-reset email has been sent to {email}."


def _tool_transfer_to_agent() -> str:
    """Return success text — actual status toggle is deferred until after the
    farewell message is sent (otherwise the bot token can't post to an 'open'
    conversation)."""
    return "Conversation transferred to a live agent."


def _toggle_conversation_open(account_id: int, conversation_id: int):
    """Toggle a Chatwoot conversation from 'pending' to 'open'."""
    url = (
        f"{CHATWOOT_URL}/api/v1/accounts/{account_id}"
        f"/conversations/{conversation_id}/toggle_status"
    )
    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_BOT_TOKEN,
    }
    try:
        resp = http_requests.post(url, json={"status": "open"}, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logging.exception("toggle_conversation_open error")


def _run_with_tools(system: str, messages: list, max_rounds: int = 5,
                    customer_email: str = None, account_id: int = None,
                    conversation_id: int = None) -> tuple:
    """Call Anthropic with tool support, looping until Claude is done.

    Returns (reply_text, transfer_requested).
    """
    transfer_requested = False
    for _ in range(max_rounds):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            system=system,
            messages=messages,
            tools=TOOLS,
            max_tokens=1024,
            temperature=0.7,
        )

        # Collect any text blocks for the final answer
        text_parts = [b.text for b in response.content if b.type == "text"]

        # If Claude didn't ask to use a tool, we're done
        if response.stop_reason != "tool_use":
            text = "\n".join(text_parts) if text_parts else ""
            return text, transfer_requested

        # Process every tool_use block in the response
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            logging.info("TOOL CALL: %s(%s)", block.name, json.dumps(block.input))
            if block.name == "transfer_to_agent":
                transfer_requested = True
            result_str = _execute_tool(block.name, block.input, customer_email,
                                       account_id=account_id, conversation_id=conversation_id)
            logging.info("TOOL RESULT: %s", result_str)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                }
            )

        # Append the assistant turn and the tool results, then loop
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Safety net — if we hit max_rounds, return whatever text we have
    text = "\n".join(text_parts) if text_parts else "I'm sorry, something went wrong."
    return text, transfer_requested


def generate_bot_response(message, customer_name=None, customer_email=None,
                          conversation_history=None, account_id=None,
                          conversation_id=None):
    """Run Pinecone similarity search + Anthropic completion for a user message."""
    results = search_similar_chats(index, message)

    context = "These are similar chat requests we have received in the past and how we responded to each:\n"
    for result in results["matches"]:
        context += "\nChat: " + result["metadata"]["chat_message"]
        context += "\nResponse: " + result["metadata"]["response"]
        context += "\n---\n"

    system = SYSTEM_PROMPT + "\n\n"

    # Include customer info so Claude can greet them by name
    if customer_name or customer_email:
        system += "Current customer info:\n"
        if customer_name:
            system += f"- Name: {customer_name}\n"
        if customer_email:
            system += f"- Email: {customer_email}\n"
        system += "\n"

    system += context

    # Build message list: use conversation history if available,
    # otherwise just the single incoming message
    if conversation_history:
        messages = list(conversation_history)
        # Ensure the latest message is included (it may already be in history
        # if Chatwoot processed it before our fetch, but append to be safe)
        if not messages or messages[-1].get("content") != message:
            messages.append({"role": "user", "content": message})
        # Claude requires messages to start with "user" role
        if messages and messages[0]["role"] != "user":
            messages = messages[1:]
    else:
        messages = [{"role": "user", "content": message}]

    reply, transfer = _run_with_tools(system, messages, customer_email=customer_email,
                                      account_id=account_id, conversation_id=conversation_id)
    return reply, transfer


def fetch_conversation_messages(account_id, conversation_id):
    """Fetch prior messages in this Chatwoot conversation.

    Returns a list of Claude-style message dicts:
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_USER_TOKEN}
    try:
        resp = http_requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json().get("payload", [])
    except Exception as e:
        logging.warning("Failed to fetch conversation history: %s", e)
        return []

    # Chatwoot messages come newest-first; reverse to chronological
    payload.sort(key=lambda m: m.get("id", 0))

    messages = []
    for msg in payload:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        msg_type = msg.get("message_type")
        # 0 = incoming (customer), 1 = outgoing (agent/bot)
        if msg_type == 0:
            role = "user"
        elif msg_type == 1:
            role = "assistant"
        else:
            continue
        # Merge consecutive same-role messages
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})

    return messages


def send_chatwoot_message(account_id, conversation_id, content):
    """Send a reply back to a Chatwoot conversation."""
    url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_BOT_TOKEN,
    }
    payload = {"content": content}
    resp = http_requests.post(url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


@MyApp.route("/webhook", methods=["POST"])
def chatwoot_webhook():
    """Handle incoming Chatwoot webhook events."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid payload"}), 400

    event = data.get("event")
    if event != "message_created":
        return jsonify({"status": "ignored", "reason": "not message_created"}), 200

    message_type = data.get("message_type")
    if message_type != "incoming":
        return jsonify({"status": "ignored", "reason": "not incoming"}), 200

    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"status": "ignored", "reason": "empty message"}), 200

    conversation = data.get("conversation") or {}
    conversation_id = conversation.get("id") or data.get("conversation", {}).get("id")
    account_id = data.get("account", {}).get("id")

    # Only respond to conversations in "pending" status (bot territory).
    # Once a conversation is "open", human agents handle it.
    conv_status = conversation.get("status")
    if conv_status and conv_status != "pending":
        return jsonify({"status": "ignored", "reason": f"conversation is {conv_status}"}), 200

    if not conversation_id or not account_id:
        _log("webhook.missing_ids", conversation_id=conversation_id, account_id=account_id)
        return jsonify({"error": "missing conversation_id or account_id"}), 400

    # Extract customer name/email from the webhook sender data
    sender = data.get("sender") or {}
    customer_name = (sender.get("name") or "").strip() or None
    customer_email = (sender.get("email") or "").strip() or None

    # Fetch prior messages so Claude sees the full conversation
    conversation_history = fetch_conversation_messages(account_id, conversation_id)

    _log("webhook.incoming", account_id=account_id, conversation_id=conversation_id,
         customer_name=customer_name, customer_email=customer_email,
         history_len=len(conversation_history), message=_trunc(content, 200))

    try:
        bot_reply, transfer_requested = generate_bot_response(
            content, customer_name=customer_name, customer_email=customer_email,
            conversation_history=conversation_history,
            account_id=account_id, conversation_id=conversation_id,
        )
        send_chatwoot_message(account_id, conversation_id, bot_reply)
        _log("webhook.replied", account_id=account_id, conversation_id=conversation_id, reply=_trunc(bot_reply, 200))

        # Toggle status AFTER sending the farewell message (bot token can't
        # post to "open" conversations)
        if transfer_requested:
            _toggle_conversation_open(account_id, conversation_id)
            _log("webhook.transferred", account_id=account_id, conversation_id=conversation_id)
    except Exception as e:
        logging.exception("webhook error")
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok"}), 200


@MyApp.route(
    "/",
    defaults={"path": ""},
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
@MyApp.route(
    "/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
)
def catch_all(path):
    """
    Catches any unrecognized route. Never pass path vars to a view
    that doesn't accept them. This function accepts the param itself.
    """

    req_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    t0 = time.time()

    norm = "/" + re.sub(r"/+", "/", (path or "").strip("/"))
    raw = request.get_data(as_text=True) or ""

    # Log every unmatched request
    _log(
        "catch_all.request",
        req_id=req_id,
        method=request.method,
        path="/" + (path or ""),
        norm=norm,
        remote_addr=request.headers.get("X-Forwarded-For") or request.remote_addr,
        user_agent=request.headers.get("User-Agent"),
        content_type=request.headers.get("Content-Type"),
        headers=dict(request.headers),
        args=request.args.to_dict(flat=True),
        raw_len=len(raw),
        raw_preview=(raw[:400] + ("…" if len(raw) > 400 else "")),
    )

    # Normalize path for fuzzy routing
    norm = "/" + re.sub(r"/+", "/", path or "").strip("/")

    # Preflight support so browser tools don't choke
    if request.method == "OPTIONS":
        resp = MyApp.response_class(status=204)
        h = resp.headers
        h["Access-Control-Allow-Origin"] = "*"
        h["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        h["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key"
        h["Access-Control-Max-Age"] = "86400"
        return resp

    # ---- Otherwise: return a rich 404 with diagnostics ----
    info = {
        "error": "not_found",
        "path": norm,
        "method": request.method,
        "hint": "Check base URL + path concatenation or OpenAPI servers/base settings.",
        "received_headers": {k: v for k, v in request.headers.items()},
    }

    # Include a small preview of the body to debug clients sending stringified JSON, etc.
    raw = request.get_data(as_text=True)
    info["raw_len"] = len(raw)
    info["raw_preview"] = raw[:300]

    # Try parse JSON even if content-type is wrong
    try:
        parsed = request.get_json(silent=True)
        if isinstance(parsed, str):
            # Some tools send JSON as a string; try a second parse
            parsed2 = json.loads(parsed)
            parsed = parsed2
        if parsed is not None:
            info["parsed_json"] = parsed
    except Exception as e:
        info["json_error"] = str(e)

    resp = jsonify(info)
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp, 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    MyApp.run(host="0.0.0.0", port=port)
