sudo python3 -m pip install transformers beautifulsoup4 warcio requests tqdm nltk questionary flask_wtf flask
sudo pip3 install torch --index-url https://download.pytorch.org/whl/cu118
mkdir fine_tuned_model
sudo python3 ./GPT2pages.py
sudo python3 ./Web_GUI.py
