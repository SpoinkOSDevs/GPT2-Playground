import torch
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertModel, DistilBertTokenizer, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from tqdm import tqdm
import os

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

# Function to filter Urban Dictionary data based on DistilBERT similarity
def filter_urban_data_with_distilbert(dataset, threshold=0.5):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Calculate DistilBERT embeddings for the definitions in the dataset
    embeddings = []
    for entry in dataset:
        definition = entry['definition']
        tokens = tokenizer(definition, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

    # Calculate similarity scores between embeddings
    similarity_matrix = torch.nn.functional.cosine_similarity(torch.tensor(embeddings), torch.tensor(embeddings))

    # Filter entries based on similarity threshold
    filtered_data = [entry for entry, sim in zip(dataset, similarity_matrix[:, 1:]) if sim > threshold]

    return filtered_data

# Function to merge datasets using a more advanced merge system
def advanced_merge_datasets(dataset1, dataset2):
    # Implement your advanced merge logic here
    # For example, you can merge datasets based on common elements or other criteria
    # This is a simplified example, modify according to your specific requirements
    merged_dataset = dataset1 + dataset2

    return merged_dataset

# Function to fine-tune GPT-2 model
def fine_tune_gpt2(input_texts, epochs=1, batch_size=4, save_path='fine_tuned_gpt2.pth'):
    # Configure GPT-2 model
    config = GPT2Config(
        n_layer=128,
        n_head=8,
        n_embd=64
    )
    gpt2_model = GPT2LMHeadModel(config)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    # Set the pad token
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Move the GPT-2 model to the appropriate device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=5e-5)

    # Tokenize the input_texts
    input_ids = gpt2_tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True)['input_ids'].to(device)
    input_ids = torch.clamp(input_ids, 0, gpt2_model.config.vocab_size - 1)

    if input_ids.numel() == 0:
        print("No valid tokens found. Please check the input_texts.")
        return

    for epoch in range(epochs):
        # Train the GPT-2 model on the entire dataset
        progress_bar = tqdm(range(0, input_ids.size(0), batch_size), desc=f'Epoch: {epoch + 1}/{epochs}')
        for i in progress_bar:
            batch_input_ids = input_ids[i:i + batch_size, :]

            # Fine-tune the GPT-2 model with the batched input
            outputs = gpt2_model(input_ids=batch_input_ids, labels=batch_input_ids)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'Loss': loss.item()})

    # Save the fine-tuned GPT-2 model
    gpt2_model.save_pretrained(save_path)

# Example: Populate the Urban Dictionary dataset, match words and definitions
urban_dataset = populate_urban_dataset(num_terms=100)

# Scrape random terms to create another dataset
random_terms = scrape_random_urban_terms(num_terms=100)
random_definitions = [scrape_urban(term) for term in random_terms]

# Create a new dataset from the random terms
random_dataset = [{'word': term, 'definition': definition} for term, definition in zip(random_terms, random_definitions)]

# Merge datasets using a more advanced merge system
merged_dataset = advanced_merge_datasets(urban_dataset, random_dataset)

# Filter merged dataset using DistilBERT for similarity
filtered_merged_dataset = filter_urban_data_with_distilbert(merged_dataset, threshold=0.7)

# Fine-tune GPT-2 with the filtered and merged dataset
fine_tune_gpt2([entry['definition'] for entry in filtered_merged_dataset], epochs=5, batch_size=4, save_path='fine_tuned_gpt2.pth')
