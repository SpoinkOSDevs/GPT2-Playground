#!/bin/bash
# Install Python Dependencies
sudo python3 -m pip install transformers beautifulsoup4 requests warcio
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo python3 -m pip install tdqm nltk questionary flask_wtf flask

# Create directory
mkdir fine_tuned_model

# Run Python scripts
sudo python3 ./GPT2pages.py
# Set the Node.js version
NODE_VERSION=18.x

# Checkout Repository
sudo git checkout v3

# Generate Package Lock
npm install --package-lock-only

# Setup Node.js
sudo curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | sudo bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
 nvm install $NODE_VERSION

# Install Dependencies
npm ci

# Build
npm run build --if-present

# Restart the terminal to apply nvm changes
echo "Restarting the terminal..."
exec bash
