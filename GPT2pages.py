import torch
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from tqdm import tqdm

# Function to scrape the Urban Dictionary for random words
def scrape_random_urban_terms(num_terms=10):
    url = f'https://www.urbandictionary.com/random.php'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all elements with the 'word' class within 'a' tags
    word_elements = soup.select('a.word')

    terms = [word_element.text.strip() for word_element in word_elements][:num_terms]
    return terms

# Function to scrape the Urban Dictionary for definitions
def scrape_urban(term):
    base_url = f"https://www.urbandictionary.com/define.php?term={term}"
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
        return None
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
        return None
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Something went wrong: {err}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    definition_element = soup.find('div', class_='meaning')
    
    if definition_element:
        definition = definition_element.text.strip()
        return definition
    else:
        return f"Couldn't find a definition for {term}."

# Function to populate the dataset with scraped words and definitions from Urban Dictionary
def populate_urban_dataset(num_terms=10):
    urban_dataset = []
    
    # Scrape random terms from Urban Dictionary
    random_terms = scrape_random_urban_terms(num_terms=num_terms)
    
    # Print out the random terms
    print("Random Urban Terms:", random_terms)

    # Retrieve the definitions for the random terms
    for term in random_terms:
        definition = scrape_urban(term)
        if definition:
            urban_dataset.append({'word': term, 'definition': definition})

    return urban_dataset

# Function to match words and definitions in the Urban Dictionary dataset
def match_words_and_definitions(dataset):
    input_texts = []
    for entry in dataset:
        word = entry['word']
        definition = entry['definition']
        input_texts.append(f"{word}: {definition}")

    # Print out the input_texts
    print("Content of input_texts:", input_texts)
    
    return input_texts

# Function to fine-tune the GPT-2 medium model on a dataset of words and definitions with batch training and progress bar
def fine_tune_gpt2_with_dataset(input_texts, epochs=1, batch_size=4, save_path='fine_tuned_model.pth'):
    if not input_texts:
        print("The input_texts list is empty. Please check the scraping logic.")
        return

    print(f"Number of input_texts: {len(input_texts)}")
    print("Sample input_texts:")
    for text in input_texts[:5]:
        print(text)

    # Configure GPT-2 model with 128 layers and 64 hidden size
    config = GPT2Config(
        n_layer=128,
        n_head=8,
        n_embd=64
    )
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    # Set the pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Move the model to the appropriate device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Tokenize the dataset and move to device
    input_ids = tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True)['input_ids'].to(device)
    input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

    if input_ids.numel() == 0:
        print("No valid tokens found. Please check the input_texts.")
        return

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


# Example: Populate the Urban Dictionary dataset, match words and definitions, then fine-tune with 5 epochs and batch size of 4
urban_dataset = populate_urban_dataset(num_terms=5)
if not urban_dataset:
    print("No dataset available. Please check the dataset population logic.")
else:
    input_texts = match_words_and_definitions(urban_dataset)
    if not input_texts:
        print("No input_texts available. Please check the dataset matching logic.")
    else:
        fine_tune_gpt2_with_dataset(input_texts, epochs=5, batch_size=4)
# Function to save the fine-tuned model and tokenizer separately
def save_model(model, tokenizer, output_path='fine_tuned_model'):
    # Save the model state dictionary
    torch.save(model.state_dict(), f'{output_path}/model_state_dict.pth')

    # Save the tokenizer's vocabulary
    tokenizer.save_pretrained(output_path)
