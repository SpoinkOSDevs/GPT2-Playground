import torch
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

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

# Function to fine-tune the GPT-2 model on the entire Urban Dictionary dataset with batch training and progress bar
def fine_tune_gpt2(epochs=1, batch_size=4):
    # Load the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Set the pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Move the model to the appropriate device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Scrape all terms from Urban Dictionary
    all_terms = scrape_all_terms()

    if all_terms:
        for epoch in range(epochs):
            # Train the model on the entire dataset in batches with a progress bar
            progress_bar = tqdm(all_terms, desc=f'Epoch: {epoch + 1}/{epochs}')
            for term in progress_bar:
                definitions = scrape_urban_dictionary(term)

                if definitions:
                    # Tokenize the definitions and move to device
                    input_ids = tokenizer(definitions, return_tensors="pt", truncation=True, padding=True)['input_ids'].to(device)

                    # Check if input_ids is empty
                    if input_ids.numel() == 0:
                        continue

                    # Ensure that input_ids are within the range of the model's vocabulary
                    input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

                    # Calculate the total number of batches
                    num_batches = (input_ids.size(0) + batch_size - 1) // batch_size

                    # Train the model in batches with a progress bar
                    for i in range(num_batches):
                        batch_start = i * batch_size
                        batch_end = min((i + 1) * batch_size, input_ids.size(0))
                        batch_input_ids = input_ids[batch_start:batch_end, :]

                        # Fine-tune the model with the batched input
                        outputs = model(batch_input_ids, labels=batch_input_ids)
                        loss = outputs.loss

                        # Backward pass and optimization
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        progress_bar.set_postfix({'Loss': loss.item()})

        # Save the fine-tuned model to a .pth file
        save_model(model, tokenizer)
    else:
        print("Error: Unable to scrape terms from Urban Dictionary.")

# Function to save the fine-tuned model to a .pth file

# Fine-tune the model on the entire Urban Dictionary dataset with progress bar
fine_tune_gpt2(epochs=5, batch_size=4)
