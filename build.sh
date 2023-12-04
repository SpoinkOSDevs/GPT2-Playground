#!/bin/bash
# Install Python Dependencies
sudo python3 -m pip install transformers beautifulsoup4 requests warcio
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo python3 -m pip install tdqm nltk questionary flask_wtf flask

# Create directory
mkdir fine_tuned_model

# Run Python scripts
sudo python3 ./GPT2pages.py
mv fine_tuned_model.pth fine_tuned_model/
sudo python3 ./gen.py
