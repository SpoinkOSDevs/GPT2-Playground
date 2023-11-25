sudo python3 -m pip install transformers beautifulsoup4 warcio requests
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
sudo python3 -m pip install tdqm nltk questionary flask_wtf flask
mkdir fine_tuned_model
sudo python3 ./GPT2.py
sudo python3 ./Web_GUI.py
