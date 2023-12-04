import torch
import requests
from bs4 import BeautifulSoup
from transformers import GPT2MediumLMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# Function to scrape the Oxford English Dictionary for words and definitions
def scrape_oed_words_and_definitions():
    url = f'https://en.oxforddictionaries.com/definition/{term}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    word_elements = soup.find_all('a', class_='word')

    words_and_definitions = []

    for word_element in word_elements:
        word = word_element.text
        definitions = scrape_oed(word)
        
        if definitions:
            words_and_definitions.append({'word': word, 'definitions': definitions})

    return words_and_definitions

# Function to fine-tune the GPT-2 medium model on a dataset of words and definitions with batch training and progress bar
def fine_tune_gpt2_with_dataset(dataset, epochs=1, batch_size=4):
    # Load the GPT-2 medium model and tokenizer
    model = GPT2MediumLMHeadModel.from_pretrained('gpt2-medium')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    # Set the pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Move the model to the appropriate device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare the dataset
    input_texts = []
    for entry in dataset:
        word = entry['word']
        definitions = entry['definitions']
        input_texts.extend([f"{word}: {definition}" for definition in definitions])

    # Tokenize the dataset and move to device
    input_ids = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True)['input_ids'].to(device)

    # Ensure that input_ids are within the range of the model's vocabulary
    input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

    for epoch in range(epochs):
        # Train the model on the entire dataset in batches with a progress bar
        progress_bar = tqdm(range(0, input_ids.size(0), batch_size), desc=f'Epoch: {epoch + 1}/{epochs}')
        for i in progress_bar:
            batch_input_ids = input_ids[i:i + batch_size, :]

            # Fine-tune the model with the batched input
            outputs = model(batch_input_ids, labels=batch_input_ids)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'Loss': loss.item()})

    # Save the fine-tuned model and tokenizer separately
    save_model(model, tokenizer, output_path='fine_tuned_model/model_state_dict.pth'')

# Scrape words and definitions from OED
word_definitions_dataset = scrape_oed_words_and_definitions()

# Fine-tune the GPT-2 medium model on the dataset of words and definitions
fine_tune_gpt2_with_dataset(word_definitions_dataset, epochs=5, batch_size=4)
