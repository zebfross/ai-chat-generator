#!/bin/bash
# Run this on the droplet (68.183.159.69) as root or with sudo
set -e

APP_DIR=/home/chatwoot/ai-chat-generator

# Clone or pull the repo
if [ ! -d "$APP_DIR" ]; then
    git clone https://github.com/zebbybobapps/ai-chat-generator.git "$APP_DIR"
    chown -R chatwoot:chatwoot "$APP_DIR"
else
    cd "$APP_DIR" && sudo -u chatwoot git pull
fi

# Install system dependencies
apt-get install -y tesseract-ocr

# Create venv and install deps
cd "$APP_DIR"
sudo -u chatwoot python3 -m venv venv
sudo -u chatwoot venv/bin/pip install --upgrade pip
sudo -u chatwoot venv/bin/pip install -r requirements.txt

# Install systemd service
cp deploy/ai-chat-bot.service /etc/systemd/system/ai-chat-bot.service
systemctl daemon-reload
systemctl enable ai-chat-bot
systemctl restart ai-chat-bot

echo ""
echo "Service status:"
systemctl status ai-chat-bot --no-pager

echo ""
echo "Next steps:"
echo "1. Create /home/chatwoot/ai-chat-generator/.env with your keys (see .env.example)"
echo "2. Add the nginx location block from deploy/nginx-bot.conf to your chat.inceptify.com server config"
echo "3. Run: nginx -t && systemctl reload nginx"
echo "4. Register the agent bot in Chatwoot with webhook URL: https://chat.inceptify.com/bot/webhook"
