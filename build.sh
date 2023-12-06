#!/bin/bash
echo 'GPT2 PLAYGROUND;'

# Install Python Dependencies
echo "Executing: sudo python3 -m pip install transformers beautifulsoup4 requests warcio argparse"
output=$(sudo python3 -m pip install transformers beautifulsoup4 requests warcio argparse 2>&1)
if [ $? -eq 0 ]; then
    echo "Command executed successfully: $output"
else
    echo "Error executing command: $output"
    exit 1
fi
clear

echo "Executing: sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
output=$(sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1)
if [ $? -eq 0 ]; then
    echo "Command executed successfully: $output"
else
    echo "Error executing command: $output"
    exit 1
fi
clear

echo "Executing: sudo python3 -m pip install tdqm nltk questionary flask_wtf flask"
output=$(sudo python3 -m pip install tdqm nltk questionary flask_wtf flask 2>&1)
if [ $? -eq 0 ]; then
    echo "Command executed successfully: $output"
else
    echo "Error executing command: $output"
    exit 1
fi
clear

mkdir fine_tuned_model
echo 'Loading . . .'

echo "Executing: sudo python3 ./GPT2pages.py"
output=$(sudo python3 ./GPT2pages.py 2>&1)
if [ $? -eq 0 ]; then
    echo "Command executed successfully: $output"
else
    echo "Error executing command: $output"
    exit 1
fi

echo 'Training the AI; this may take a while . . .'

mv fine_tuned_model.pth fine_tuned_model/
clear

echo 'Starting Webserver . . .'
sleep 0.3
echo 'Brushing the Cat . .'
sleep 2
echo 'Fine-tuning the Cogs and gears .'
rm ./README.md

echo 'Done!'
echo "Executing: sudo python3 ./gen.py"
output=$(sudo python3 ./gen.py 2>&1)
if [ $? -eq 0 ]; then
    echo "Command executed successfully: $output"
else
    echo "Error executing command: $output"
    exit 1
fi
