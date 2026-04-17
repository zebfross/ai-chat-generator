#!/usr/bin/env bash
# Fire fake Chatwoot webhooks at the local Flask app to test CRM sync.
#
# Usage:
#   ./scripts/test_crm_webhook.sh created
#   ./scripts/test_crm_webhook.sh resolved
#   ./scripts/test_crm_webhook.sh both
#
# Env overrides:
#   URL=http://localhost:5050/webhook
#   NAME="Alice Test"
#   EMAIL=alice@test.com
#   PHONE=+15557771234
#   CONV=99999

set -euo pipefail

URL="${URL:-http://localhost:5050/webhook}"
NAME="${NAME:-Test User}"
EMAIL="${EMAIL:-test@example.com}"
PHONE="${PHONE:-+15555551234}"
CONV="${CONV:-99999}"

which="${1:-}"
if [[ "$which" != "created" && "$which" != "resolved" && "$which" != "both" ]]; then
  echo "Usage: $0 {created|resolved|both}" >&2
  exit 1
fi

fire() {
  local event="$1" status="$2"
  local body
  body=$(cat <<JSON
{
  "event": "$event",
  "account": {"id": 3},
  "conversation": {
    "id": $CONV,
    "inbox_id": 1,
    "status": "$status",
    "meta": {"sender": {"name": "$NAME", "email": "$EMAIL", "phone_number": "$PHONE"}}
  },
  "inbox": {"name": "Website", "channel_type": "Channel::WebWidget"}
}
JSON
)
  echo
  echo "→ POST $URL  event=$event  status=$status"
  curl -sS -X POST "$URL" \
    -H 'Content-Type: application/json' \
    -d "$body" \
    -w '\n← HTTP %{http_code}\n'
}

if [[ "$which" == "created" || "$which" == "both" ]]; then
  fire "conversation_created" "open"
fi
if [[ "$which" == "resolved" || "$which" == "both" ]]; then
  fire "conversation_status_changed" "resolved"
fi
