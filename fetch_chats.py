"""Fetch all Tawk.to chats and export as JSON for Pinecone ingestion."""
import json
import time
import requests

API_KEY = "ak00eb0cd8b39d2f48e6ab9b06096ef4035f"
PROPERTY_ID = "6080c09d62662a09efc0e484"
BASE_URL = "https://api.tawk.to/v1"
PAGE_SIZE = 50

def fetch_chats():
    all_chats = []
    offset = 0

    while True:
        resp = requests.post(
            f"{BASE_URL}/chat.list",
            auth=(API_KEY, ""),
            json={
                "propertyId": PROPERTY_ID,
                "size": PAGE_SIZE,
                "offset": offset,
                "startDate": "2022-02-01T00:00:00Z",
                "endDate": "2026-03-04T00:00:00Z",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("ok"):
            print(f"API error: {data}")
            break

        chats = data.get("data", [])
        if not chats:
            break

        all_chats.extend(chats)
        print(f"Fetched {len(all_chats)} / {data.get('total', '?')} chats")

        if len(all_chats) >= data.get("total", 0):
            break

        offset += PAGE_SIZE
        time.sleep(0.5)  # rate limit

    return all_chats


def extract_conversations(chats):
    """Extract visitor/agent exchange pairs from chat messages.

    Walks through each chat chronologically, pairing each visitor message
    with the next agent response. This captures the full conversation,
    not just the opening greeting.
    """
    import re

    conversations = []

    def is_junk(text):
        t = text.lower().strip()
        if len(t) < 8:
            return True
        if t in ("hi", "hello", "hey", "help", "hi there", "hi there,",
                 "hello!", "hey!", "ok", "okay", "thanks", "thank you",
                 "yes", "no", "sure", "alright", "got it", "cool"):
            return True
        # Form fills
        if t.startswith("name :") or ("email :" in t and "phone :" in t):
            return True
        # Just a phone number
        if re.match(r"^[\d\s\-\(\)\+]+$", t):
            return True
        return False

    def is_non_answer(text):
        t = text.lower().strip()
        if len(t) < 15:
            return True
        # Strip greeting prefix
        substance = re.sub(r"^(hi|hello|hey)\s+\w+[,!.\s]*", "", t).strip()
        # Strip trailing "how can I help"
        substance = re.sub(
            r"(how can i help|how may i help|how can i assist|what can i do for you)[?\s!]*$",
            "", substance
        ).strip()
        if len(substance) < 10:
            return True
        # Pure deflections
        deflections = [
            r"^i('d| would) (love|be happy|be glad) to help[!.\s]*$",
            r"^i can (definitely |certainly )?help[!.\s]*$",
            r"^(nice|great|awesome) to meet you[!.\s]*$",
            r"^thanks for the info[!.\s]*$",
            r"^(great|ok|sure|absolutely)[!.\s]*$",
        ]
        for p in deflections:
            if re.match(p, substance):
                return True
        return False

    for chat in chats:
        messages = chat.get("messages", [])
        created = chat.get("createdOn", "")

        # Build ordered list of (sender_type, text)
        ordered = []
        for msg in messages:
            if msg.get("type") != "msg":
                continue
            sender_type = msg.get("sender", {}).get("t", "")
            text = msg.get("msg", "").strip()
            if not text:
                continue
            if text.startswith("Welcome to our site"):
                continue
            if sender_type in ("v", "a"):
                ordered.append((sender_type, text))

        # Pair each visitor message with the next agent response
        for i, (stype, text) in enumerate(ordered):
            if stype != "v":
                continue
            if is_junk(text):
                continue
            # Find next agent message
            for j in range(i + 1, len(ordered)):
                if ordered[j][0] == "a":
                    agent_text = ordered[j][1]
                    if not is_non_answer(agent_text):
                        conversations.append({
                            "chat_message": text,
                            "response": agent_text,
                            "timestamp": created,
                        })
                    break

    return conversations


if __name__ == "__main__":
    print("Fetching chats from Tawk.to...")
    chats = fetch_chats()
    print(f"\nTotal chats fetched: {len(chats)}")

    conversations = extract_conversations(chats)
    print(f"Conversations with visitor+agent pairs: {len(conversations)}")

    output_file = "tawk_conversations.json"
    with open(output_file, "w") as f:
        json.dump(conversations, f, indent=2)
    print(f"Saved to {output_file}")

    # Show a few samples
    for c in conversations[:3]:
        print(f"\n  Q: {c['chat_message'][:80]}")
        print(f"  A: {c['response'][:80]}")
