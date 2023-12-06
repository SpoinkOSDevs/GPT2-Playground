#!/bin/bash
echo 'GPT2 PLAYGROUND;'
# Install Python Dependencies
sudo python3 -m pip install transformers beautifulsoup4 requests warcio argparse
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo python3 -m pip install tdqm nltk questionary flask_wtf flask
# Create directory
mkdir fine_tuned_model
clear
echo 'DEPS INSTALLED'
# Run Python scripts
sudo python3 ./GPT2pages.py --num_layers 254 --num_heads 16 --num_embeddings 1024 --epochs 10 --batch_size 8
mv fine_tuned_model.pth fine_tuned_model/
clear
echo 'Starting Webserver . . .'
sudo python3 ./gen.py
