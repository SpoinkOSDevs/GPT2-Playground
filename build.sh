#!/bin/bash
echo 'GPT2 PLAYGROUND;'
# Install Python Dependencies
sudo python3 -m pip install transformers beautifulsoup4 requests warcio argparse
clear
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
clear
sudo python3 -m pip install tdqm nltk questionary flask_wtf flask
clear
mkdir fine_tuned_model
echo 'Loading . . .'
sudo python3 ./GPT2pages.py
echo 'Training the AI this may take a while . . .'
mv fine_tuned_model.pth fine_tuned_model/
clear
echo 'Starting Webserver . . .'
sleep 0.3
echo 'Brushing the Cat . .'
sleep 2
echo 'Fine tuning the Cogs and gears .'
rm ./README.md
Echo 'Done!'
sudo python3 ./gen.py
