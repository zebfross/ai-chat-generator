import collections
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

from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS

try:
    import yaml  # pip install pyyaml
except ImportError:
    yaml = None
MyApp = Flask(__name__)

# Optional: set up root logger if you haven't already (Heroku reads stdout)
logging.basicConfig(level=logging.INFO)

# In-memory ring buffer for recent requests (viewable at /debug/logs)
TOOL_LOG = collections.deque(maxlen=50)
# Thread-local trace for the current request
_current_trace = None


def _trace_event(event_type, **data):
    """Append an event to the current request trace."""
    global _current_trace
    if _current_trace is not None:
        _current_trace["events"].append({
            "type": event_type,
            "ts": datetime.utcnow().isoformat(),
            **data,
        })

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


@MyApp.route("/debug/logs")
def debug_logs():
    """Return recent request traces for debugging."""
    limit = request.args.get("limit", 10, type=int)
    traces = list(TOOL_LOG)[-limit:]
    return jsonify(traces)


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
CHATWOOT_ACCOUNT_ID = int(os.environ.get("CHATWOOT_ACCOUNT_ID", "3"))
_email_ids_raw = os.environ.get("EMAIL_INBOX_IDS", os.environ.get("EMAIL_INBOX_ID", ""))
EMAIL_INBOX_IDS = set()
for _id in _email_ids_raw.split(","):
    _id = _id.strip()
    if _id:
        try:
            EMAIL_INBOX_IDS.add(int(_id))
        except ValueError:
            pass

_sms_ids_raw = os.environ.get("SMS_INBOX_IDS", "")
SMS_INBOX_IDS = set()
for _id in _sms_ids_raw.split(","):
    _id = _id.strip()
    if _id:
        try:
            SMS_INBOX_IDS.add(int(_id))
        except ValueError:
            pass

SYSTEM_PROMPT = (
    "You are a TradelineWorks.com support chat agent. "
    "This is a live chat — keep responses very short, casual, and conversational. "
    "Use 1-4 sentences max. No bullet points, no headers, no numbered lists unless showing search results. "
    "Reply like a friendly human support agent would in a chat window.\n\n"

    "HOW TRADELINES WORK:\n"
    "A tradeline is being added as an authorized user on someone else's established credit card. "
    "After purchase, the customer uploads their information and is added to the card within 1-3 days. "
    "The tradeline generally shows up on their credit report 4-6 weeks after the report date "
    "(the report date is shown on the tradeline purchase page). "
    "Customers pay monthly to stay on the tradeline as long as they want, with a 3-month minimum. "
    "When they are removed, the tradeline simply appears as a closed account on their credit report. "
    "Do NOT claim that payment history, account age, or other benefits remain after removal. "
    "We do NOT guarantee credit score increases.\n\n"

    "POLICIES:\n"
    "We do not allow CPNs — they are illegal. We only work with legitimate Social Security Numbers.\n"
    "We do not guarantee seller payouts — payouts depend on the buyer staying on the tradeline and making successful payments.\n"
    "When a customer asks for a phone call, do not offer to call them directly. "
    "Instead, share this scheduling link so they can book a call: "
    "https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ0wNCGZTSpEY2fI6wPXSQU9jEr-hPa9hK8nkWTqQhJHWFkaVIRePrl_Xvlw5rkYiWpwAkEqCmmw\n\n"

    "TOOLS:\n"
    "You have tools to search tradelines, look up orders, reset passwords, "
    "cancel orders, check seller payouts, and search the company knowledgebase. "
    "Use the search_knowledge tool when a customer asks about processes, policies, "
    "or anything you're not confident answering from the info above. "
    "Use these tools before transferring to a human agent.\n\n"
    "If you cannot help the customer after using the available tools, or they explicitly ask for a human agent, "
    "use the transfer_to_agent tool to hand the conversation to a live agent.\n\n"

    "I'll give you previous chat requests we've received and how we responded to help you. "
    "Try to stay with the provided style and content.\n\n"
    "When showing tradeline results, preserve the markdown links exactly as provided by the tool."
)

EMAIL_SYSTEM_PROMPT = (
    "You are a TradelineWorks.com customer support representative responding via email. "
    "Write complete, self-contained responses — do not assume back-and-forth like a live chat.\n\n"
    "Guidelines:\n"
    "- Use a professional, friendly tone appropriate for email.\n"
    "- Start with a greeting (e.g. \"Hi [Name],\") and end with a sign-off "
    "(e.g. \"Best regards,\\nTradelineWorks Support\").\n"
    "- Write in plain text — no markdown formatting, no bullet points with dashes.\n"
    "- Provide thorough, helpful answers in paragraph form.\n"
    "- If you cannot help the customer, or they explicitly ask for a human agent, "
    "use the transfer_to_agent tool to hand the conversation to a live agent.\n"
    "- When a customer asks for a phone call, do not offer to call them directly. "
    "Instead, share this scheduling link so they can book a call: "
    "https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ0wNCGZTSpEY2fI6wPXSQU9jEr-hPa9hK8nkWTqQhJHWFkaVIRePrl_Xvlw5rkYiWpwAkEqCmmw\n"
    "- Use your tools (search tradelines, look up orders, search knowledgebase, etc.) "
    "to find accurate information before responding.\n\n"
    "I'll give you previous chat requests we've received and how we responded for reference."
)

# ---------------------------------------------------------------------------
# Tool definitions — sent with every Anthropic request so Claude knows
# what actions it can take.  Add new tools here.
# ---------------------------------------------------------------------------
TW_BASE_URL = os.environ.get("TW_BASE_URL", "").rstrip("/")
TW_BOT_API_KEY = os.environ.get("TW_BOT_API_KEY", "")
TW_VERIFY_SSL = not any(h in TW_BASE_URL for h in ("localhost", ".local", "127.0.0.1"))
CLICKUP_API_TOKEN = os.environ.get("CLICKUP_API_TOKEN", "")
CLICKUP_LIST_ID = os.environ.get("CLICKUP_LIST_ID", "901708695881")

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
            "Use this when a customer asks about the status of their order(s), "
            "payment dates, posting dates, or any details about their current orders."
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
        "name": "cancel_order",
        "description": (
            "Cancel a customer's tradeline order. "
            "Use this when a customer wants to cancel a specific order. "
            "Requires the customer's email and the order ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The customer's email address.",
                },
                "order_id": {
                    "type": "number",
                    "description": "The order ID to cancel.",
                },
            },
            "required": ["email", "order_id"],
        },
    },
    {
        "name": "seller_payouts",
        "description": (
            "Look up a seller's payout history and pending payouts. "
            "Use this when a seller asks about their payouts, earnings, or payments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The seller's email address.",
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
    {
        "name": "create_task",
        "description": (
            "Create a task in the team's task management system (ClickUp). "
            "Use this when a customer request needs follow-up by the team, "
            "such as escalations, issues that can't be resolved in chat, "
            "refund requests, or anything that requires manual action by staff."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A brief title for the task.",
                },
                "description": {
                    "type": "string",
                    "description": "Details about what needs to be done, including any relevant context from the conversation.",
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority level: 1=urgent, 2=high, 3=normal, 4=low. Default to 3 if unsure.",
                    "enum": [1, 2, 3, 4],
                },
            },
            "required": ["title", "description"],
        },
    },
    {
        "name": "search_knowledge",
        "description": (
            "Read a knowledgebase article by its ID. The list of available "
            "articles and their IDs is in your system prompt. Use this when "
            "a customer asks about processes, policies, or anything you're "
            "not confident answering from your instructions alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_id": {
                    "type": "number",
                    "description": "The ID of the knowledgebase article to read.",
                },
            },
            "required": ["article_id"],
        },
    },
    {
        "name": "create_verification",
        "description": (
            "Generate a Stripe Identity verification link for a customer. "
            "Use this when a customer needs to verify their identity "
            "(e.g., for a new order or re-verification). Returns a URL "
            "the customer can visit to complete ID verification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The customer's email address.",
                },
                "allowed_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Document types to accept. Options: 'driving_license', "
                        "'id_card', 'passport'. Defaults to all three."
                    ),
                },
                "require_selfie": {
                    "type": "boolean",
                    "description": (
                        "Whether to require a matching selfie. Default true."
                    ),
                },
                "require_id_number": {
                    "type": "boolean",
                    "description": (
                        "Whether to require SSN/ID number. Default true."
                    ),
                },
            },
            "required": ["email"],
        },
    },
    {
        "name": "get_user",
        "description": (
            "Look up a TradelineWorks user by email or phone number. "
            "Returns their name, phone number, roles, and registration date. "
            "Use this to get contact info or determine if someone is a buyer, seller, or salesperson."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The user's email address.",
                },
                "phone": {
                    "type": "string",
                    "description": "The user's phone number (any format).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_calendar_events",
        "description": (
            "Fetch raw calendar events for a date range. "
            "Use this to check staff schedules and answer questions like "
            "'is Porsha available tomorrow?' by interpreting event titles "
            "(e.g. 'Porsha Off' means unavailable, 'Alex 1-3' means Alex works 1-3pm)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Defaults to today.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to fetch (1-14). Defaults to 1.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_calendar_availability",
        "description": (
            "Get available 30-minute appointment slots during business hours "
            "(9am-5pm ET, weekdays only). Use this when a customer wants to "
            "schedule a call or meeting and needs to see open time slots."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Defaults to today.",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to check (1-14). Defaults to 3.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "create_calendar_event",
        "description": (
            "Create a calendar event and send invites. "
            "Use this to book a meeting or call with a customer. "
            "Always confirm the time with the customer before creating the event."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Event title (e.g. 'Call with John Smith').",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in YYYY-MM-DD HH:MM format (Eastern Time).",
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration in minutes (15-120). Defaults to 30.",
                },
                "attendee": {
                    "type": "string",
                    "description": "Attendee email address to send an invite to.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional event description or notes.",
                },
            },
            "required": ["summary", "start_time"],
        },
    },
]


def _tw_headers():
    return {"X-Bot-Api-Key": TW_BOT_API_KEY}


def _fetch_knowledge_titles() -> str:
    """Fetch knowledgebase article titles at startup for system prompt context."""
    try:
        resp = requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/knowledge",
            headers=_tw_headers(),
            verify=TW_VERIFY_SSL,
            timeout=10,
        )
        if resp.ok:
            articles = resp.json()
            if articles:
                lines = ["KNOWLEDGEBASE ARTICLES (use search_knowledge tool with the article ID to read):"]
                for a in articles:
                    lines.append(f"  - [ID {a['id']}] {a['title']}")
                return "\n".join(lines)
    except Exception:
        pass
    return ""


_KNOWLEDGE_TITLES = _fetch_knowledge_titles() if TW_BASE_URL else ""


def _lookup_user_info(email: str = None, phone: str = None) -> dict | None:
    """Look up a user's full profile (name, email, phone, roles) via the WP endpoint."""
    if not email and not phone:
        return None
    params = {}
    if email:
        params["email"] = email
    if phone:
        params["phone"] = phone
    try:
        resp = requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/user",
            params=params,
            headers=_tw_headers(),
            timeout=5,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return None


def _execute_tool(name: str, tool_input: dict, customer_email: str = None,
                   account_id: int = None, conversation_id: int = None) -> str:
    """Dispatch a tool call and return a plain-text result string."""
    if name == "search_tradelines":
        return _tool_search_tradelines(tool_input)
    if name == "order_lookup":
        return _tool_order_lookup(tool_input, customer_email)
    if name == "reset_password":
        return _tool_reset_password(tool_input, customer_email)
    if name == "cancel_order":
        return _tool_cancel_order(tool_input, customer_email)
    if name == "seller_payouts":
        return _tool_seller_payouts(tool_input, customer_email)
    if name == "transfer_to_agent":
        return _tool_transfer_to_agent()
    if name == "create_task":
        return _tool_create_task(tool_input, conversation_id, account_id)
    if name == "search_knowledge":
        return _tool_search_knowledge(tool_input)
    if name == "create_verification":
        return _tool_create_verification(tool_input, customer_email)
    if name == "get_user":
        return _tool_get_user(tool_input)
    if name == "get_calendar_events":
        return _tool_get_calendar_events(tool_input)
    if name == "get_calendar_availability":
        return _tool_get_calendar_availability(tool_input)
    if name == "create_calendar_event":
        return _tool_create_calendar_event(tool_input)
    return f"Unknown tool: {name}"


def _tool_search_knowledge(tool_input: dict) -> str:
    """Fetch a knowledgebase article by ID from the TradelineWorks API."""
    article_id = tool_input.get("article_id")
    if not article_id:
        return "article_id is required. Check the article list in your system prompt."
    try:
        resp = requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/knowledge",
            params={"id": int(article_id)},
            headers=_tw_headers(),
            verify=TW_VERIFY_SSL,
            timeout=10,
        )
        if not resp.ok:
            return f"Knowledge base lookup failed (HTTP {resp.status_code})."
        data = resp.json()
        return f"**{data.get('title', '')}**\n\n{data.get('content', '')}"
    except Exception as exc:
        return f"Knowledge base lookup error: {exc}"


def _tool_create_verification(tool_input: dict, customer_email: str = None) -> str:
    """Create a Stripe Identity verification session via the TW API."""
    email = tool_input.get("email") or customer_email
    if not email:
        return "Error: email is required."
    payload = {"email": email}
    if "allowed_types" in tool_input:
        payload["allowed_types"] = tool_input["allowed_types"]
    if "require_selfie" in tool_input:
        payload["require_selfie"] = tool_input["require_selfie"]
    if "require_id_number" in tool_input:
        payload["require_id_number"] = tool_input["require_id_number"]
    if "return_url" in tool_input:
        payload["return_url"] = tool_input["return_url"]
    try:
        resp = requests.post(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/create-verification",
            json=payload,
            headers=_tw_headers(),
            verify=TW_VERIFY_SSL,
            timeout=15,
        )
        if not resp.ok:
            return f"Verification link creation failed (HTTP {resp.status_code}): {resp.text}"
        data = resp.json()
        return f"Verification link created: {data.get('verification_url', 'N/A')}"
    except Exception as exc:
        return f"Verification link error: {exc}"


def _tool_get_user(tool_input: dict) -> str:
    """Look up a TradelineWorks user by email or phone via the TW API."""
    email = tool_input.get("email", "").strip()
    phone = tool_input.get("phone", "").strip()
    if not email and not phone:
        return "Error: email or phone is required."
    info = _lookup_user_info(email=email or None, phone=phone or None)
    if not info:
        return "No user found matching that email or phone number."
    lines = []
    if info.get("name"):
        lines.append(f"Name: {info['name']}")
    if info.get("email"):
        lines.append(f"Email: {info['email']}")
    if info.get("phone"):
        lines.append(f"Phone: {info['phone']}")
    if info.get("role"):
        lines.append(f"Role: {info['role']}")
    if info.get("registered"):
        lines.append(f"Registered: {info['registered']}")
    return "\n".join(lines) if lines else str(info)


def _tool_get_calendar_events(tool_input: dict) -> str:
    """Fetch raw calendar events for a date range via the TW API."""
    params = {}
    if tool_input.get("date"):
        params["date"] = tool_input["date"]
    if tool_input.get("days"):
        params["days"] = tool_input["days"]
    try:
        resp = requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/calendar-events",
            params=params,
            headers=_tw_headers(),
            verify=TW_VERIFY_SSL,
            timeout=10,
        )
        if not resp.ok:
            return f"Calendar events lookup failed (HTTP {resp.status_code})."
        data = resp.json()
        if not data.get("events"):
            return "No calendar events found for the requested date range."
        lines = []
        for event in data["events"]:
            lines.append(f"- {event.get('start', '')} — {event.get('summary', '')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Calendar events error: {exc}"


def _tool_get_calendar_availability(tool_input: dict) -> str:
    """Get available appointment slots via the TW API."""
    params = {}
    if tool_input.get("date"):
        params["date"] = tool_input["date"]
    if tool_input.get("days"):
        params["days"] = tool_input["days"]
    try:
        resp = requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/calendar-availability",
            params=params,
            headers=_tw_headers(),
            verify=TW_VERIFY_SSL,
            timeout=10,
        )
        if not resp.ok:
            return f"Availability lookup failed (HTTP {resp.status_code})."
        data = resp.json()
        if not data.get("slots"):
            return "No available slots found for the requested date range."
        lines = []
        for slot in data["slots"]:
            lines.append(f"- {slot.get('start', '')} to {slot.get('end', '')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Availability lookup error: {exc}"


def _tool_create_calendar_event(tool_input: dict) -> str:
    """Create a calendar event via the TW API."""
    summary = tool_input.get("summary")
    start_time = tool_input.get("start_time")
    if not summary or not start_time:
        return "Error: summary and start_time are required."
    payload = {"summary": summary, "start_time": start_time}
    if tool_input.get("duration"):
        payload["duration"] = tool_input["duration"]
    if tool_input.get("attendee"):
        payload["attendee"] = tool_input["attendee"]
    if tool_input.get("description"):
        payload["description"] = tool_input["description"]
    try:
        resp = requests.post(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/calendar-event",
            json=payload,
            headers=_tw_headers(),
            verify=TW_VERIFY_SSL,
            timeout=15,
        )
        if not resp.ok:
            return f"Failed to create calendar event (HTTP {resp.status_code}): {resp.text}"
        data = resp.json()
        return f"Calendar event created: {data.get('summary', summary)} at {data.get('start', start_time)}"
    except Exception as exc:
        return f"Calendar event creation error: {exc}"


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
            f"- [{c['name']} – ${int(float(c['limit'])):,} limit]({c['url']}) | "
            f"Age: {c['age_months']} months | Price: ${c['price']} | "
            f"Stock: {c['stock_remaining']}"
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


def _tool_cancel_order(tool_input: dict, customer_email: str = None) -> str:
    """Cancel an order via the TW API."""
    email = tool_input.get("email", "").strip()
    order_id = tool_input.get("order_id")
    if not email or not order_id:
        return "Error: both email and order_id are required."
    if not customer_email:
        return "Error: cannot cancel order — customer email is not verified for this session."
    if email.lower() != customer_email.lower():
        return f"Error: for security, I can only cancel orders for the current customer ({customer_email})."

    try:
        resp = http_requests.post(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/cancel-order",
            json={"email": email, "order_id": int(order_id)},
            headers=_tw_headers(),
            timeout=15,
            verify=TW_VERIFY_SSL,
        )
        if resp.status_code == 404:
            return f"Order {order_id} not found for {email}."
        if resp.status_code == 400:
            try:
                err = resp.json()
                reason = err.get("code") or err.get("message") or resp.text[:300]
            except Exception:
                reason = resp.text[:300]
            return f"Cannot cancel order {order_id}: {reason}"
        if not resp.ok:
            body = resp.text[:300]
            logging.error("cancel_order %s: %s", resp.status_code, body)
            return f"Error cancelling order: {body}"
        result = resp.json()
    except Exception as e:
        logging.exception("cancel_order error")
        return f"Error cancelling order: {e}"

    return result.get("message", f"Order {order_id} has been cancelled.")


def _tool_seller_payouts(tool_input: dict, customer_email: str = None) -> str:
    """Look up seller payouts via the TW API."""
    email = tool_input.get("email", "").strip()
    if not email:
        return "Error: no email address provided."
    if not customer_email:
        return "Error: cannot look up payouts — customer email is not verified for this session."
    if email.lower() != customer_email.lower():
        return f"Error: for security, I can only look up payouts for the current customer ({customer_email})."

    try:
        resp = http_requests.get(
            f"{TW_BASE_URL}/wp-json/tw/v1/bot/seller-payouts",
            params={"email": email},
            headers=_tw_headers(),
            timeout=15,
            verify=TW_VERIFY_SSL,
        )
        if resp.status_code == 404:
            return f"No seller account found for {email}."
        if not resp.ok:
            body = resp.text[:300]
            logging.error("seller_payouts %s: %s", resp.status_code, body)
            return f"Error looking up payouts: {body}"
        data = resp.json()
    except Exception as e:
        logging.exception("seller_payouts error")
        return f"Error looking up payouts: {e}"

    if not data:
        return f"No payout information found for {email}."

    return data.get("message", str(data))


def _get_online_agents() -> list:
    """Return list of agents with availability_status == 'online'."""
    try:
        url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/agents"
        headers = {"api_access_token": CHATWOOT_USER_TOKEN}
        resp = http_requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        agents = resp.json()
        return [a for a in agents if a.get("availability_status") == "online"]
    except Exception as e:
        logging.exception("_get_online_agents error")
        return []


def _tool_transfer_to_agent() -> str:
    """Check if agents are online before transferring.

    Returns success text if agents are available, otherwise instructs the bot
    to let the customer know someone will follow up during business hours.
    """
    online = _get_online_agents()
    if not online:
        return (
            "No agents are currently online. Let the customer know someone "
            "will get back to them during business hours (Mon-Fri 9am-6pm EST). "
            "Do NOT use the transfer_to_agent tool again."
        )
    return "Conversation transferred to a live agent."


def _tool_create_task(tool_input: dict, conversation_id: int = None,
                      account_id: int = None) -> str:
    """Create a ClickUp task and post a whisper in Chatwoot with the link."""
    if not CLICKUP_API_TOKEN:
        return "Error: ClickUp integration is not configured."

    title = tool_input.get("title", "").strip()
    description = tool_input.get("description", "").strip()
    priority = tool_input.get("priority", 3)

    if not title:
        return "Error: task title is required."

    full_description = description
    if conversation_id and account_id:
        full_description += (
            f"\n\n---\nSource: Chatwoot conversation #{conversation_id}\n"
            f"Conversation: {CHATWOOT_URL}/app/accounts/{account_id}/conversations/{conversation_id}"
        )

    try:
        resp = http_requests.post(
            f"https://api.clickup.com/api/v2/list/{CLICKUP_LIST_ID}/task",
            json={"name": title, "description": full_description, "priority": priority},
            headers={"Authorization": CLICKUP_API_TOKEN, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        task = resp.json()
        task_url = task.get("url", "")
    except Exception as e:
        logging.exception("create_task error")
        return f"Error creating task: {e}"

    # Post whisper (private note) in Chatwoot with the task link
    if conversation_id and account_id and task_url:
        try:
            whisper_url = (
                f"{CHATWOOT_URL}/api/v1/accounts/{account_id}"
                f"/conversations/{conversation_id}/messages"
            )
            http_requests.post(
                whisper_url,
                json={
                    "content": f"\U0001f4cb **Task created:** [{title}]({task_url})\n\n_Priority: {priority}_",
                    "message_type": "outgoing",
                    "private": True,
                },
                headers={
                    "Content-Type": "application/json",
                    "api_access_token": CHATWOOT_USER_TOKEN,
                },
                timeout=10,
            )
        except Exception:
            logging.exception("create_task whisper error")

    return f"Task created: {title} — {task_url}"


def _toggle_conversation_open(account_id: int, conversation_id: int):
    """Toggle a Chatwoot conversation from 'pending' to 'open'."""
    url = (
        f"{CHATWOOT_URL}/api/v1/accounts/{account_id}"
        f"/conversations/{conversation_id}/toggle_status"
    )
    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_USER_TOKEN,
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
    _trace_event("anthropic_request", system=system,
                 messages=[{"role": m["role"], "content": str(m.get("content", ""))[:500]}
                           for m in messages[-10:]],
                 tool_names=[t["name"] for t in TOOLS])

    for round_num in range(max_rounds):
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

        # Summarize response content for trace
        response_summary = []
        for block in response.content:
            if block.type == "text":
                response_summary.append({"type": "text", "text": block.text[:300]})
            elif block.type == "tool_use":
                response_summary.append({"type": "tool_use", "name": block.name, "input": block.input})

        _trace_event("anthropic_response", round=round_num,
                     stop_reason=response.stop_reason,
                     content=response_summary,
                     usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens})

        # If Claude didn't ask to use a tool, we're done
        if response.stop_reason != "tool_use":
            text = "\n".join(text_parts) if text_parts else ""
            return text, transfer_requested

        # Process every tool_use block in the response
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_str = _execute_tool(block.name, block.input, customer_email,
                                       account_id=account_id, conversation_id=conversation_id)
            if block.name == "transfer_to_agent" and not result_str.startswith("No agents"):
                transfer_requested = True
            _trace_event("tool_executed", tool=block.name, input=block.input,
                         result=result_str[:500])
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
                          sender_phone=None, conversation_history=None,
                          account_id=None, conversation_id=None, inbox_id=None):
    """Run Pinecone similarity search + Anthropic completion for a user message."""
    results = search_similar_chats(index, message)

    _trace_event("pinecone_search", query=message,
                 matches=[{"score": round(m["score"], 3), "chat": m["metadata"]["chat_message"][:100]}
                          for m in results["matches"]])

    context = "These are similar chat requests we have received in the past and how we responded to each:\n"
    for result in results["matches"]:
        context += "\nChat: " + result["metadata"]["chat_message"]
        context += "\nResponse: " + result["metadata"]["response"]
        context += "\n---\n"

    # Enrich customer info from WP user lookup
    user_info = _lookup_user_info(customer_email, sender_phone)
    if user_info:
        customer_name = customer_name or user_info.get("name")
        customer_email = customer_email or user_info.get("email")

    is_email = inbox_id in EMAIL_INBOX_IDS
    base_prompt = EMAIL_SYSTEM_PROMPT if is_email else SYSTEM_PROMPT
    system = base_prompt + "\n\n"

    if _KNOWLEDGE_TITLES:
        system += _KNOWLEDGE_TITLES + "\n\n"

    # Include customer info so Claude can greet them by name
    if customer_name or customer_email or sender_phone:
        system += "Current customer info:\n"
        if customer_name:
            system += f"- Name: {customer_name}\n"
        if customer_email:
            system += f"- Email: {customer_email}\n"
        role = (user_info or {}).get("role") or "unknown (likely buyer)"
        system += f"- Role: {role.capitalize()}\n"
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


def send_chatwoot_private_message(account_id, conversation_id, content):
    """Send a private note (whisper) visible only to agents."""
    url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    headers = {
        "Content-Type": "application/json",
        "api_access_token": CHATWOOT_USER_TOKEN,
    }
    payload = {"content": content, "private": True}
    resp = http_requests.post(url, json=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def generate_handoff_summary(conversation_history):
    """Generate a brief summary of the conversation for the receiving agent."""
    if not conversation_history:
        return "No conversation history available."

    # Build a condensed version of the conversation for summarization
    convo_text = ""
    for msg in conversation_history:
        role = "Customer" if msg["role"] == "user" else "Bot"
        convo_text += f"{role}: {msg['content']}\n"

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            system=(
                "Summarize this customer support conversation for a human agent "
                "who is taking over. Focus on: what the customer needs, what was already tried/discussed, "
                "and any important details (email, order numbers, etc). Be concise."
            ),
            messages=[{"role": "user", "content": convo_text}],
            max_tokens=200,
            temperature=0,
        )
        return response.content[0].text
    except Exception as e:
        logging.exception("generate_handoff_summary error")
        return "Bot was unable to generate a summary of this conversation."


def _is_sender_excluded(sender_id: str) -> bool:
    """Check if a sender matches the exclusion list."""
    excluded = os.environ.get("EXCLUDED_SENDERS", "[]")
    try:
        excluded_list = json.loads(excluded)
    except Exception:
        excluded_list = []
    return any(exc.lower() in sender_id.lower() for exc in excluded_list if exc)


@MyApp.route("/webhook", methods=["POST"])
def chatwoot_webhook():
    """Unified bot webhook — handles chat responses, email drafts, and task analysis."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid payload"}), 400

    event = data.get("event")

    # Handle new conversation → create contact in Pipedrive/SendGrid (no transcript yet)
    if event == "conversation_created":
        return _handle_conversation_created(data)

    # Handle conversation resolved → push transcript to Pipedrive/SendGrid
    if event == "conversation_status_changed":
        return _handle_conversation_status_changed(data)

    if event != "message_created":
        return jsonify({"status": "ignored", "reason": "not message_created"}), 200

    message_type = data.get("message_type")
    if message_type != "incoming":
        return jsonify({"status": "ignored", "reason": "not incoming"}), 200

    if data.get("private"):
        return jsonify({"status": "ignored", "reason": "private note"}), 200

    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"status": "ignored", "reason": "empty message"}), 200

    conversation = data.get("conversation") or {}
    conversation_id = conversation.get("id") or data.get("conversation", {}).get("id")
    account_id = data.get("account", {}).get("id")
    inbox_id = conversation.get("inbox_id")
    inbox = data.get("inbox") or {}
    inbox_name = inbox.get("name") or "Unknown"

    # Skip if a human agent is assigned — let them handle it.
    assignee = conversation.get("assignee")
    if assignee:
        return jsonify({"status": "ignored", "reason": "assigned to agent"}), 200

    if not conversation_id or not account_id:
        _log("webhook.missing_ids", conversation_id=conversation_id, account_id=account_id)
        return jsonify({"error": "missing conversation_id or account_id"}), 400

    # Extract customer name/email from the webhook sender data
    sender = data.get("sender") or {}
    customer_name = (sender.get("name") or "").strip() or None
    customer_email = (sender.get("email") or "").strip() or None
    sender_phone = (sender.get("phone_number") or "").strip()
    sender_id = customer_email or sender_phone or customer_name or ""

    is_excluded = _is_sender_excluded(sender_id)
    is_email = inbox_id in EMAIL_INBOX_IDS
    is_sms = inbox_id in SMS_INBOX_IDS
    is_draft_only = is_email or is_sms

    # Start a trace for this request
    global _current_trace
    _current_trace = {
        "ts": datetime.utcnow().isoformat(),
        "conversation_id": conversation_id,
        "account_id": account_id,
        "inbox_id": inbox_id,
        "inbox_name": inbox_name,
        "is_email": is_email,
        "is_sms": is_sms,
        "message": content,
        "sender": {k: sender.get(k) for k in ("id", "name", "email", "phone_number", "identifier")},
        "customer_name": customer_name,
        "customer_email": customer_email,
        "is_excluded": is_excluded,
        "events": [],
    }

    try:
        if is_draft_only:
            _handle_draft_message(content, conversation_id, account_id, inbox_id,
                                  inbox_name, customer_name, customer_email, sender_phone,
                                  sender_id, is_excluded, is_email=is_email)
        else:
            _handle_chat_message(content, conversation_id, account_id, inbox_id,
                                 inbox_name, customer_name, customer_email, sender_phone,
                                 sender_id, is_excluded)
    except Exception as e:
        logging.exception("webhook error")
        _trace_event("error", error=str(e))
        _current_trace["error"] = str(e)
        TOOL_LOG.append(_current_trace)
        _current_trace = None
        return jsonify({"error": str(e)}), 500

    TOOL_LOG.append(_current_trace)
    _current_trace = None

    return jsonify({"status": "ok"}), 200


def _handle_chat_message(content, conversation_id, account_id, inbox_id,
                          inbox_name, customer_name, customer_email, sender_phone,
                          sender_id, is_excluded):
    """Handle chat/SMS messages: respond to customer + run task analysis."""
    conversation_history = fetch_conversation_messages(account_id, conversation_id)
    _trace_event("history_fetched", message_count=len(conversation_history),
                 messages=[{"role": m["role"], "content": m["content"][:200]} for m in conversation_history[-10:]])

    bot_reply, transfer_requested = generate_bot_response(
        content, customer_name=customer_name, customer_email=customer_email,
        sender_phone=sender_phone, conversation_history=conversation_history,
        account_id=account_id, conversation_id=conversation_id,
        inbox_id=inbox_id,
    )
    send_chatwoot_message(account_id, conversation_id, bot_reply)
    _trace_event("reply_sent", reply=bot_reply[:500])

    # Bot handled the message — clear action-required
    _taskbot_remove_label(account_id, conversation_id, "action-required")

    if transfer_requested:
        summary = generate_handoff_summary(conversation_history)
        _toggle_conversation_open(account_id, conversation_id)
        send_chatwoot_private_message(
            account_id, conversation_id,
            f"**Bot Summary:** {summary}"
        )
        # Handoff means a human still needs to act
        _taskbot_add_label(account_id, conversation_id, "action-required")
        _trace_event("handoff", summary=summary[:500])

    # Run TaskBot analysis (unless sender is excluded)
    if not is_excluded:
        _run_taskbot_analysis(content, conversation_id, account_id,
                              customer_name, sender_id, inbox_name, conversation_history)


def _handle_draft_message(content, conversation_id, account_id, inbox_id,
                           inbox_name, customer_name, customer_email, sender_phone,
                           sender_id, is_excluded, is_email=True):
    """Handle draft-only inboxes (email/SMS): draft reply as whisper + run task analysis."""
    if is_excluded:
        logging.info("[RILEY] Skipping excluded sender: %s", sender_id)
        return

    channel_type = "email" if is_email else "SMS"
    logging.info("[RILEY] Processing %s from %s in %s", channel_type, sender_id, inbox_name)

    conversation_history = fetch_conversation_messages(account_id, conversation_id)
    _trace_event("history_fetched", message_count=len(conversation_history))

    # Build system prompt with Pinecone context
    results = search_similar_chats(index, content)
    context = "These are similar chat requests we have received in the past and how we responded to each:\n"
    for result in results["matches"]:
        context += "\nChat: " + result["metadata"]["chat_message"]
        context += "\nResponse: " + result["metadata"]["response"]
        context += "\n---\n"

    base_prompt = EMAILBOT_SYSTEM_PROMPT if is_email else SMS_DRAFT_PROMPT
    system = base_prompt + "\n\n"
    # Enrich customer info from WP user lookup
    user_info = _lookup_user_info(customer_email, sender_phone)
    if user_info:
        customer_name = customer_name or user_info.get("name")
        customer_email = customer_email or user_info.get("email")

    if customer_name or customer_email or sender_phone:
        system += "Current customer info:\n"
        if customer_name:
            system += f"- Name: {customer_name}\n"
        if customer_email:
            system += f"- Email: {customer_email}\n"
        role = (user_info or {}).get("role") or "unknown (likely buyer)"
        system += f"- Role: {role.capitalize()}\n"
        system += "\n"
    system += context

    # Build message list from conversation history
    if conversation_history:
        messages = list(conversation_history)
        if not messages or messages[-1].get("content") != content:
            messages.append({"role": "user", "content": content})
        if messages and messages[0]["role"] != "user":
            messages = messages[1:]
    else:
        messages = [{"role": "user", "content": content}]

    # Generate draft with tools
    draft, tools_used = _emailbot_run_with_tools(system, messages, customer_email=customer_email,
                                                  account_id=account_id, conversation_id=conversation_id)

    if draft:
        whisper = "**Draft Reply:**\n\n" + draft
        if tools_used:
            tool_lines = []
            for t in tools_used:
                tool_lines.append(f"- `{t['name']}({t['input']})` → {t['result'][:200]}")
            whisper += "\n\n---\n**Tools used:**\n" + "\n".join(tool_lines)
        send_chatwoot_private_message(account_id, conversation_id, whisper)
        _trace_event("email_draft_sent", draft=draft[:500])
        # Draft generated — clear action-required until next incoming message
        _taskbot_remove_label(account_id, conversation_id, "action-required")

    # Run TaskBot analysis
    _run_taskbot_analysis(content, conversation_id, account_id,
                          customer_name, sender_id, inbox_name, conversation_history)


def _run_taskbot_analysis(content, conversation_id, account_id,
                           customer_name, sender_id, inbox_name, conversation_history=None):
    """Run TaskBot analysis — add labels, priority, and whisper if action needed."""
    if conversation_history is None:
        conversation_history = fetch_conversation_messages(account_id, conversation_id)

    context_lines = []
    for m in conversation_history[-10:]:
        role = "Customer" if m["role"] == "user" else "Agent"
        context_lines.append(f"{role}: {m['content']}")
    context = "\n".join(context_lines)

    try:
        analysis = _taskbot_analyze(content, context, customer_name or sender_id, inbox_name)
    except Exception:
        logging.exception("[RILEY] TaskBot analysis error")
        return

    if not analysis or not analysis.get("needs_action"):
        return

    priority_map = {1: "urgent", 2: "high", 3: "medium", 4: "low"}
    priority = priority_map.get(analysis.get("priority", 3), "medium")

    _taskbot_add_label(account_id, conversation_id, "action-required")
    _taskbot_set_priority(account_id, conversation_id, priority)

    summary = analysis.get("summary", "")
    whisper = f"**Action Required** ({priority}): {summary}"
    send_chatwoot_private_message(account_id, conversation_id, whisper)
    _trace_event("taskbot_flagged", priority=priority, summary=summary[:200])
    logging.info("[RILEY] Flagged conversation %s: %s (%s)", conversation_id, summary[:80], priority)



# ---------------------------------------------------------------------------
# CRM Integration — Pipedrive + SendGrid on conversation resolved
# ---------------------------------------------------------------------------

def _sync_conversation_to_crm(data, include_transcript, event_name):
    """Shared CRM sync path — forwards contact + optional transcript to the WP LHC endpoint."""
    conversation = data.get("conversation") or data
    conversation_id = conversation.get("id")
    account_id = data.get("account", {}).get("id")

    if not conversation_id or not account_id:
        return jsonify({"error": "missing conversation_id or account_id"}), 400

    meta = conversation.get("meta") or {}
    sender = meta.get("sender") or data.get("sender") or {}
    name = (sender.get("name") or "").strip()
    email = (sender.get("email") or "").strip()
    phone = (sender.get("phone_number") or "").strip()

    if not email and not phone:
        logging.info("[CRM] No email or phone for conversation %s, skipping", conversation_id)
        return jsonify({"status": "ignored", "reason": "no contact info"}), 200

    sender_id = email or phone or name
    if _is_sender_excluded(sender_id):
        logging.info("[CRM] Skipping excluded sender: %s", sender_id)
        return jsonify({"status": "ignored", "reason": "excluded sender"}), 200

    inbox = data.get("inbox") or {}
    channel_type = inbox.get("channel_type") or ""
    if "Internal" in channel_type:
        return jsonify({"status": "ignored", "reason": "internal channel"}), 200

    payload = {
        "event": event_name,
        "chat": {
            "id": conversation_id,
            "nick": name,
            "email": email,
            "phone": phone,
            "url": f"{CHATWOOT_URL}/app/accounts/{account_id}/conversations/{conversation_id}",
        },
    }
    if include_transcript:
        payload["messages"] = _build_chat_messages(account_id, conversation_id)

    try:
        resp = http_requests.post(
            f"{TW_BASE_URL}/wp-json/cmfs/v1/lhc",
            json=payload,
            verify=TW_VERIFY_SSL,
            timeout=20,
        )
        if resp.ok:
            logging.info("[CRM] Synced conversation %s (%s) to LHC endpoint", conversation_id, event_name)
        else:
            logging.error("[CRM] LHC sync failed (HTTP %s): %s", resp.status_code, resp.text[:300])
    except Exception:
        logging.exception("[CRM] LHC sync error")

    return jsonify({"status": "ok"}), 200


def _handle_conversation_created(data):
    """When a new conversation is opened, create contact in CRM (no transcript)."""
    conversation = data.get("conversation") or data
    conversation_id = conversation.get("id")
    logging.info("[CRM] Conversation %s created — creating CRM contact", conversation_id)
    return _sync_conversation_to_crm(data, include_transcript=False, event_name="chat.chat_started")


def _handle_conversation_status_changed(data):
    """When a conversation is resolved, push transcript to CRM."""
    conversation = data.get("conversation") or data
    status = (conversation.get("status") or "").lower()

    if status != "resolved":
        return jsonify({"status": "ignored", "reason": f"status is {status}, not resolved"}), 200

    conversation_id = conversation.get("id")
    logging.info("[CRM] Conversation %s resolved — syncing transcript to CRM", conversation_id)
    return _sync_conversation_to_crm(data, include_transcript=True, event_name="chat.chat_closed")

    return jsonify({"status": "ok", "synced": True}), 200


def _build_chat_messages(account_id, conversation_id):
    """Return a list of {time, name, msg} dicts for the LHC endpoint."""
    messages = fetch_conversation_messages(account_id, conversation_id)
    out = []
    for m in messages or []:
        out.append({
            "name": "Customer" if m["role"] == "user" else "Agent",
            "msg": m.get("content", ""),
            "time": "",
        })
    return out


def _taskbot_add_label(account_id: int, conversation_id: int, label: str):
    """Add a label to a conversation, preserving existing labels."""
    try:
        url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/labels"
        headers = {"Content-Type": "application/json", "api_access_token": CHATWOOT_USER_TOKEN}
        resp = http_requests.get(url, headers=headers, timeout=10)
        existing = resp.json().get("payload", []) if resp.ok else []
        if label not in existing:
            existing.append(label)
            http_requests.post(url, json={"labels": existing}, headers=headers, timeout=10)
    except Exception:
        logging.exception("[TASKBOT] Error adding label")


def _taskbot_remove_label(account_id: int, conversation_id: int, label: str):
    """Remove a label from a conversation if present."""
    try:
        url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}/labels"
        headers = {"Content-Type": "application/json", "api_access_token": CHATWOOT_USER_TOKEN}
        resp = http_requests.get(url, headers=headers, timeout=10)
        existing = resp.json().get("payload", []) if resp.ok else []
        if label in existing:
            existing.remove(label)
            http_requests.post(url, json={"labels": existing}, headers=headers, timeout=10)
    except Exception:
        logging.exception("[TASKBOT] Error removing label")


def _taskbot_set_priority(account_id: int, conversation_id: int, priority: str):
    """Set priority on a conversation."""
    try:
        url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conversation_id}"
        headers = {"Content-Type": "application/json", "api_access_token": CHATWOOT_USER_TOKEN}
        http_requests.patch(url, json={"priority": priority}, headers=headers, timeout=10)
    except Exception:
        logging.exception("[TASKBOT] Error setting priority")


def _taskbot_analyze(message: str, context: str, sender_name: str, inbox_name: str) -> dict:
    """Use Claude to decide if a conversation needs action."""
    prompt = (
        "You are a business assistant for TradelineWorks, a tradeline company. "
        "You analyze incoming customer messages and decide if the conversation needs attention from the team.\n\n"
        "Flag as action required when the message contains:\n"
        "- A specific request or question that needs follow-up\n"
        "- A complaint or escalation\n"
        "- A time-sensitive matter\n"
        "- An HR/admin request\n"
        "- A request to purchase or sell tradelines\n\n"
        "Do NOT flag:\n"
        "- Simple greetings or 'thank you' messages\n"
        "- Messages that are already being handled in the conversation\n"
        "- Automated/system messages\n"
        "- Spam or irrelevant messages\n\n"
        f"Inbox: {inbox_name}\n"
        f"Sender: {sender_name}\n\n"
        f"Recent conversation context:\n{context}\n\n"
        f"New message to analyze:\n{message}\n\n"
        "Respond with JSON only:\n"
        '{"needs_action": true/false, '
        '"summary": "Brief description of what action is needed", '
        '"priority": 1-4 (1=urgent, 2=high, 3=medium, 4=low)}'
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )

    text = response.content[0].text
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not json_match:
        return None

    return json.loads(json_match.group(0))


EMAILBOT_SYSTEM_PROMPT = (
    "You are a TradelineWorks.com customer support representative drafting an email response. "
    "Write a complete, professional email reply that an agent can review and send.\n\n"
    "Guidelines:\n"
    "- Use a professional, friendly tone appropriate for email.\n"
    "- Start with a greeting (e.g. \"Hi [Name],\") and end with a sign-off "
    "(e.g. \"Best regards,\\nTradelineWorks Support\").\n"
    "- Write in plain text — no markdown formatting, no bullet points with dashes.\n"
    "- Provide thorough, helpful answers in paragraph form.\n"
    "- Use your tools to look up real data (orders, tradelines, etc.) before responding.\n"
    "- If you cannot help the customer, say so in the draft and suggest the agent handle it manually.\n"
    "- Do NOT use the transfer_to_agent tool — this is a draft, not a live conversation.\n\n"

    "HOW TRADELINES WORK:\n"
    "A tradeline is being added as an authorized user on someone else's established credit card. "
    "After purchase, the customer uploads their information and is added to the card within 1-3 days. "
    "The tradeline generally shows up on their credit report 4-6 weeks after the report date "
    "(the report date is shown on the tradeline purchase page). "
    "Customers pay monthly to stay on the tradeline as long as they want, with a 3-month minimum. "
    "When they are removed, the tradeline simply appears as a closed account on their credit report. "
    "Do NOT claim that payment history, account age, or other benefits remain after removal. "
    "We do NOT guarantee credit score increases. Never promise a refund or imply that our company "
    "made any mistakes.\n\n"

    "POLICIES:\n"
    "We do not allow CPNs — they are illegal. We only work with legitimate Social Security Numbers.\n"
    "When a customer asks for a phone call, do not offer to call them directly. "
    "Instead, share this scheduling link so they can book a call: "
    "https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ0wNCGZTSpEY2fI6wPXSQU9jEr-hPa9hK8nkWTqQhJHWFkaVIRePrl_Xvlw5rkYiWpwAkEqCmmw\n\n"

    "I'll give you previous chat requests we've received and how we responded for reference."
)

SMS_DRAFT_PROMPT = (
    "You are a TradelineWorks.com support agent drafting an SMS reply for agent review. "
    "Write a short, casual text message — 1-3 sentences max. "
    "No greetings or sign-offs needed for texts. Keep it conversational.\n\n"

    "HOW TRADELINES WORK:\n"
    "A tradeline is being added as an authorized user on someone else's established credit card. "
    "After purchase, the customer uploads their information and is added to the card within 1-3 days. "
    "The tradeline generally shows up on their credit report 4-6 weeks after the report date. "
    "Customers pay monthly with a 3-month minimum. "
    "We do NOT guarantee credit score increases.\n\n"

    "POLICIES:\n"
    "We do not allow CPNs — they are illegal. We only work with legitimate Social Security Numbers.\n"
    "When a customer asks for a phone call, do not offer to call them directly. "
    "Instead, share this scheduling link so they can book a call: "
    "https://calendar.google.com/calendar/u/0/appointments/schedules/AcZssZ0wNCGZTSpEY2fI6wPXSQU9jEr-hPa9hK8nkWTqQhJHWFkaVIRePrl_Xvlw5rkYiWpwAkEqCmmw\n\n"

    "Use your tools to look up real data before responding. "
    "Do NOT use the transfer_to_agent tool — this is a draft, not a live conversation.\n\n"

    "I'll give you previous chat requests we've received and how we responded for reference."
)

# Tools available to draft bots — same as Riley but without transfer_to_agent
EMAILBOT_TOOLS = [t for t in TOOLS if t["name"] != "transfer_to_agent"]



def _emailbot_run_with_tools(system: str, messages: list, max_rounds: int = 5,
                              customer_email: str = None, account_id: int = None,
                              conversation_id: int = None) -> tuple:
    """Call Anthropic with EmailBot tools, looping until done.

    Returns (draft_text, tools_used) where tools_used is a list of
    {"name": ..., "input": ..., "result": ...} dicts.
    """
    tools_used = []
    text_parts = []

    for round_num in range(max_rounds):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            system=system,
            messages=messages,
            tools=EMAILBOT_TOOLS,
            max_tokens=1024,
            temperature=0.7,
        )

        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            return "\n".join(text_parts), tools_used

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tool_use in tool_uses:
            result = _execute_tool(
                tool_use.name, tool_use.input,
                customer_email=customer_email,
                account_id=account_id,
                conversation_id=conversation_id,
            )
            tools_used.append({
                "name": tool_use.name,
                "input": json.dumps(tool_use.input),
                "result": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return ("\n".join(text_parts) if text_parts else None), tools_used


@MyApp.route("/ocr", methods=["POST"])
def ocr_image():
    """Accept an image file and return OCR text via AWS Textract."""
    import boto3
    from PIL import Image, ImageOps
    import io

    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send as multipart form field 'file'."}), 400

    file = request.files["file"]
    try:
        # Auto-rotate based on EXIF orientation and convert to bytes
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        textract = boto3.client("textract", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        response = textract.detect_document_text(Document={"Bytes": image_bytes})

        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block["BlockType"] == "LINE"
        ]
        return jsonify({"text": "\n".join(lines), "lines": lines})
    except Exception as e:
        logging.exception("OCR error")
        return jsonify({"error": str(e)}), 500


# ── Bot Response Review Tool ────────────────────────────────────────────

REVIEW_KEY = os.environ.get("REVIEW_KEY", "")

import sqlite3

_REVIEW_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review.db")


def _review_db():
    """Get a SQLite connection (creates table on first call)."""
    conn = sqlite3.connect(_REVIEW_DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reviewed_messages "
        "(message_id INTEGER PRIMARY KEY, reviewed_at TEXT)"
    )
    return conn


# iPhone tapback reactions: Liked "...", Loved "...", etc.
_TAPBACK_RE = re.compile(
    r'^(Liked|Loved|Disliked|Laughed at|Emphasized|Questioned)\s+"', re.IGNORECASE
)


def _check_review_key():
    """Return True if the request carries a valid review key, else False."""
    if not REVIEW_KEY:
        return False
    return request.args.get("key") == REVIEW_KEY


@MyApp.route("/review")
def review_page():
    if not _check_review_key():
        return "Unauthorized", 401
    return render_template("review.html", review_key=REVIEW_KEY)


@MyApp.route("/review/messages")
def review_messages():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    page = request.args.get("page", 1, type=int)
    after = request.args.get("after", "")  # ISO date string
    account_id = CHATWOOT_ACCOUNT_ID

    # Fetch conversations from Chatwoot (sorted newest first)
    params = {"page": page}
    url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations"
    headers = {"api_access_token": CHATWOOT_USER_TOKEN}

    try:
        resp = http_requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        conversations = resp.json().get("data", {}).get("payload", [])
    except Exception as e:
        logging.exception("review: failed to fetch conversations")
        return jsonify({"error": str(e)}), 500

    results = []
    for conv in conversations:
        conv_id = conv.get("id")
        inbox = conv.get("inbox", {}) or {}
        inbox_name = inbox.get("name") or conv.get("meta", {}).get("channel", "Unknown")

        # Fetch messages for this conversation
        msgs_url = f"{CHATWOOT_URL}/api/v1/accounts/{account_id}/conversations/{conv_id}/messages"
        try:
            msgs_resp = http_requests.get(msgs_url, headers=headers, timeout=10)
            msgs_resp.raise_for_status()
            messages = msgs_resp.json().get("payload", [])
        except Exception:
            continue

        # Sort chronologically
        messages.sort(key=lambda m: m.get("id", 0))

        # Find bot responses: private whispers (drafts) or outgoing from bot
        for i, msg in enumerate(messages):
            msg_type = msg.get("message_type")  # 0=incoming, 1=outgoing, 2=activity
            is_private = msg.get("private", False)
            content = (msg.get("content") or "").strip()
            created = msg.get("created_at")

            if not content:
                continue

            # Skip iPhone tapback reactions
            if _TAPBACK_RE.match(content):
                continue

            # Apply date filter
            if after and created:
                try:
                    msg_time = datetime.utcfromtimestamp(created)
                    after_time = datetime.fromisoformat(after.replace("Z", "+00:00").replace("+00:00", ""))
                    if msg_time < after_time:
                        continue
                except Exception:
                    pass

            # Bot responses are either:
            # 1. Private notes (whispers) — drafts for email/SMS
            # 2. Outgoing messages from the bot agent (msg_type=1 with no human sender)
            is_bot_whisper = is_private and msg_type == 1
            sender = msg.get("sender") or {}
            is_bot_outgoing = (msg_type == 1 and not is_private
                               and sender.get("type") == "agent_bot")

            if not (is_bot_whisper or is_bot_outgoing):
                continue

            # Find the preceding customer message (skip tapback reactions)
            customer_message = ""
            for j in range(i - 1, -1, -1):
                prev = messages[j]
                prev_content = (prev.get("content") or "").strip()
                if (prev.get("message_type") == 0 and prev_content
                        and not _TAPBACK_RE.match(prev_content)):
                    customer_message = prev_content
                    break

            # Skip if the customer message that triggered this was a tapback
            if customer_message and _TAPBACK_RE.match(customer_message):
                continue

            results.append({
                "customer_message": customer_message,
                "bot_response": content,
                "conversation_id": conv_id,
                "message_id": msg.get("id"),
                "created_at": created,
                "inbox_id": conv.get("inbox_id"),
                "inbox_name": inbox_name,
                "is_private": is_private,
                "conversation_url": f"{CHATWOOT_URL}/app/accounts/{account_id}/conversations/{conv_id}",
            })

    # Sort newest first
    results.sort(key=lambda r: r.get("created_at") or 0, reverse=True)

    return jsonify({"messages": results, "page": page})


@MyApp.route("/review/context")
def review_context():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "query param required"}), 400

    results = search_similar_chats(index, query, top_k=5)
    items = []
    for match in results.get("matches", []):
        items.append({
            "score": round(match["score"], 4),
            "question": match["metadata"].get("chat_message", ""),
            "answer": match["metadata"].get("response", ""),
        })

    return jsonify({"context": items})


@MyApp.route("/review/system-prompt")
def review_system_prompt():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    inbox_id = request.args.get("inbox_id", type=int)
    is_email = inbox_id in EMAIL_INBOX_IDS if inbox_id else False
    is_sms = inbox_id in SMS_INBOX_IDS if inbox_id else False
    is_draft = is_email or is_sms

    if is_email:
        prompt = EMAILBOT_SYSTEM_PROMPT
        channel = "email"
    elif is_sms:
        prompt = SMS_DRAFT_PROMPT
        channel = "sms"
    elif is_draft:
        prompt = EMAILBOT_SYSTEM_PROMPT
        channel = "draft"
    else:
        prompt = SYSTEM_PROMPT
        channel = "chat"

    return jsonify({"system_prompt": prompt, "channel": channel})


@MyApp.route("/review/conversation-history")
def review_conversation_history():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    conversation_id = request.args.get("conversation_id", type=int)
    if not conversation_id:
        return jsonify({"error": "conversation_id required"}), 400

    messages = fetch_conversation_messages(CHATWOOT_ACCOUNT_ID, conversation_id)
    return jsonify({"messages": messages})


@MyApp.route("/review/correct", methods=["POST"])
def review_correct():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()

    if not question or not answer:
        return jsonify({"error": "question and answer required"}), 400

    store_chat(index, question, answer, datetime.utcnow().isoformat())
    return jsonify({"status": "saved"})


@MyApp.route("/review/reviewed")
def review_reviewed():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    conn = _review_db()
    rows = conn.execute("SELECT message_id FROM reviewed_messages").fetchall()
    conn.close()
    return jsonify({"reviewed": [r[0] for r in rows]})


@MyApp.route("/review/mark-reviewed", methods=["POST"])
def review_mark_reviewed():
    if not _check_review_key():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    message_id = data.get("message_id")
    if not message_id:
        return jsonify({"error": "message_id required"}), 400

    conn = _review_db()
    conn.execute(
        "INSERT OR IGNORE INTO reviewed_messages (message_id, reviewed_at) VALUES (?, ?)",
        (message_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


# ── Catch-all (must be last) ───────────────────────────────────────────

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
