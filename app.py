import json
import logging
import os
import re
import sys
import time
import uuid
from urllib.parse import parse_qs

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


from openai import OpenAI
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

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=os.environ["openai_key"])
pc = Pinecone(api_key=os.environ["pinecone_key"])
timer("create pinecone db")
# Define the Pinecone index name and embedding model
db_index_name = "chat-history"
embedding_model = "text-embedding-ada-002"

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


# Function to generate embeddings using OpenAI
def get_openai_embedding(text):
    response = openai.embeddings.create(input=text, model=embedding_model)
    return response["data"][0]["embedding"]


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
    results = search_similar_chats(index, query)
    timer("find similar chats")
    history = (
        """Can you pretend to be a TradelineWorks.com support chat agent?  Keep responses short and concise.  I'll give you previous chat requests we've received and how we responded to help you.  Try to stay with the provided style and content.

Please respond to the current chat.

The current chat request is: """
        + query
        + """

These are similar chat requests we have received in the past and how we responded to each: """
    )

    for result in results["matches"]:
        history += "\nChat: " + result["metadata"]["chat_message"]
        history += "\nResponse:" + result["metadata"]["response"]
        # history += "Timestamp:" + result['metadata']['timestamp']
        history += "\n---\n"

    # print(history)
    # exit()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or use "gpt-3.5-turbo" for cheaper and faster results
        messages=[
            {"role": "system", "content": history},
            {"role": "user", "content": query},
        ],
        max_tokens=100,  # Limit the response length
        temperature=0.7,  # Adjust creativity (0.0 = deterministic, 1.0 = very creative)
    )

    # Print the response
    # print("ChatGPT response:")
    timer("chatgpt response")
    return response.choices[0].message.content


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
                "id": m.get("id", ""),
                "message": msg,
                "response": resp,
                "timestamp": timestamp,  # may be None (allowed by schema)
                "score": float(m.get("score", 0.0)),
                "source": "pinecone",
                "metadata": passthrough,
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

    lines = []
    for it in items:
        q = (it.get("message") or "").strip().replace("\n", " ")
        a = (it.get("response") or "").strip().replace("\n", " ")
        if not q or not a:
            continue
        lines.append(f"- Q: {q}\n  A: {a}")
    summary = "\n\n".join(lines)

    return {
        "results": items,
        "summary": summary
    }, 200


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
