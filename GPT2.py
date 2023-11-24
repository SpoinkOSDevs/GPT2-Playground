import torch
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from tqdm import tqdm
import questionary

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the DistilBERT model and tokenizer for text classification
classifier_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2")
classifier_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2")

# Set the pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Move the GPT-2 model to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize GPT-2 model optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Function to filter out inappropriate content using DistilBERT
def filter_inappropriate_content(text):
    # Tokenize the text
    tokens = classifier_tokenizer(text, return_tensors="pt")

    # Make a prediction
    outputs = classifier_model(**tokens)

    # Check if the model predicts the text as inappropriate
    if outputs.logits.argmax() == 1:  # 1 represents inappropriate class in the SST-2 model
        return '*' * len(text)  # Replace with your preferred masking

    return text

# Function to scrape Urban Dictionary for definitions
def scrape_urban_dictionary(term):
    url = f'https://www.urbandictionary.com/define.php?term={term}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    definitions = soup.find_all('div', class_='meaning')

    if definitions:
        return [defn.get_text(separator='\n', strip=True) for defn in definitions]
    else:
        return None

# Function to scrape all terms from Urban Dictionary
def scrape_all_terms():
    url = 'https://www.urbandictionary.com/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    terms = soup.find_all('a', class_='word')

    if terms:
        return [term.text for term in terms]
    else:
        return None

# Function to save the fine-tuned model and tokenizer separately
def save_model(model, tokenizer, output_path='fine_tuned_model'):
    # Save the model state dictionary
    torch.save(model.state_dict(), f'{output_path}/model_state_dict.pth')

    # Save the tokenizer's vocabulary
    tokenizer.save_pretrained(output_path)

# Function to get user configuration
def get_user_config():
    epochs = questionary.text("Enter the number of epochs:").ask()
    batch_size = questionary.text("Enter the batch size:").ask()
    return int(epochs), int(batch_size)

# Config menu
def config_menu():
    print("=== Model Training Configuration ===")
    epochs, batch_size = get_user_config()
    fine_tune_gpt2(epochs=epochs, batch_size=batch_size)

# Main screen
def main_screen():
    choices = ["Configure Model", "Exit"]
    option = questionary.select("Choose an option:", choices=choices).ask()
    
    if option == "Configure Model":
        config_menu()
    elif option == "Exit":
        print("Goodbye!")
        exit()

# Function to fine-tune the GPT-2 model on the entire Urban Dictionary dataset with batch training and progress bar
def fine_tune_gpt2(epochs=1, batch_size=4):
    # ... (Your previous fine-tuning code)

# Main function
def main():
    while True:
        main_screen()

if __name__ == "__main__":
    main()
