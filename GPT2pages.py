import argparse
import torch
import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from tqdm import tqdm
import numpy as np
import string

# Function to scrape the Urban Dictionary for random words (excluding non-Latin characters)
def scrape_random_urban_terms(num_terms=10):
    url = f'https://www.urbandictionary.com/random.php'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all elements with the 'word' class within 'a' tags
    word_elements = soup.select('a.word')

    terms = [word_element.text.strip() for word_element in word_elements][:num_terms]

    # Exclude terms with non-Latin characters
    latin_terms = [term for term in terms if all(char in string.ascii_letters for char in term)]
<<<<<<< HEAD

=======
    
>>>>>>> 79011dcd19fb8f56e6f3d54873b363bed4900767
    return latin_terms

# Function to scrape the Urban Dictionary for definitions
def scrape_urban(term):
    base_url = f"https://www.urbandictionary.com/define.php?term={term}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        if response.status_code == 404:
            print(f"Term not found: {term}")
            return None
        else:
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

<<<<<<< HEAD
# Function to fine-tune GPT-2 model with dynamic configuration
def fine_tune_gpt2_dynamic_config(input_texts, num_layers, num_heads, num_embeddings, epochs=1, batch_size=4):
=======
# Function to filter Urban Dictionary data based on DistilBERT similarity
def filter_urban_data_with_distilbert(dataset, threshold=0.7):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Calculate DistilBERT embeddings for the definitions in the dataset
    embeddings = []
    max_length = 0  # Track the maximum length for padding

    for entry in dataset:
        definition = entry['definition']
        tokens = tokenizer(definition, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**tokens)

        # Use mean pooling over the first dimension (tokens)
        mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(mean_embedding)

        # Track the maximum length for padding
        max_length = max(max_length, mean_embedding.shape[0])

    # Pad the embeddings to have the same length
    embeddings = [np.pad(embedding, (0, max_length - embedding.shape[0])) for embedding in embeddings]

    # Stack the padded embeddings into a single NumPy array
    embeddings = np.stack(embeddings, axis=0)

    # Calculate similarity scores between embeddings
    similarity_matrix = torch.nn.functional.cosine_similarity(torch.tensor(embeddings), torch.tensor(embeddings), dim=0)

    # Filter entries based on similarity threshold
    filtered_data = [entry for entry, sim in zip(dataset, similarity_matrix.tolist()) if sim > threshold]

    return filtered_data

# Function to fine-tune GPT-2 model
def fine_tune_gpt2(input_texts, epochs=1, batch_size=4, save_path='fine_tuned_gpt2.pth'):
>>>>>>> 79011dcd19fb8f56e6f3d54873b363bed4900767
    config = GPT2Config(
        n_layer=num_layers,
        n_head=num_heads,
        n_embd=num_embeddings
    )
    gpt2_model = GPT2LMHeadModel(config)

    # Set the pad token
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_model.to(device)

    optimizer = torch.optim.AdamW(gpt2_model.parameters(), lr=5e-5)

<<<<<<< HEAD
    # Scraping random terms
    urban_dataset = scrape_random_urban_terms(num_terms=100)
    random_definitions = [scrape_urban(term) for term in urban_dataset]
    random_dataset = [{'word': term, 'definition': definition} for term, definition in zip(urban_dataset, random_definitions)]

    # Tokenize and preprocess for fine-tuning
    input_texts = [entry['definition'] for entry in random_dataset]
=======
>>>>>>> 79011dcd19fb8f56e6f3d54873b363bed4900767
    input_ids = gpt2_tokenizer(input_texts, return_tensors="pt", truncation=True, padding=True)['input_ids'].to(device)
    input_ids = torch.clamp(input_ids, 0, gpt2_model.config.vocab_size - 1)

    if input_ids.numel() == 0:
        print("No valid tokens found. Please check the input_texts.")
        return None

    # Fine-tune the model
    for epoch in range(epochs):
        progress_bar = tqdm(range(0, input_ids.size(0), batch_size), desc=f'Epoch: {epoch + 1}/{epochs}')
        for i in progress_bar:
            batch_input_ids = input_ids[i:i + batch_size, :]

            outputs = gpt2_model(batch_input_ids, labels=batch_input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'Loss': loss.item()})

<<<<<<< HEAD
    return gpt2_model

# Command-line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with dynamic configuration.")
    parser.add_argument("--num_layers", type=int, default=64, help="Number of layers for GPT-2 model.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for GPT-2 model.")
    parser.add_argument("--num_embeddings", type=int, default=32, help="Number of embeddings for GPT-2 model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for fine-tuning.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Example usage with dynamic configuration from command-line arguments
    trained_model = fine_tune_gpt2_dynamic_config(
        [entry['definition'] for entry in random_dataset],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_embeddings=args.num_embeddings,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Save the trained model
    torch.save(trained_model.state_dict(), 'fine_tuned_model/fine_tuned_model.pth')
=======
    torch.save(gpt2_model.state_dict(), save_path)

# Example usage
urban_dataset = scrape_random_urban_terms(num_terms=100)

random_definitions = [scrape_urban(term) for term in urban_dataset]
random_dataset = [{'word': term, 'definition': definition} for term, definition in zip(urban_dataset, random_definitions)]

filtered_urban_dataset = filter_urban_data_with_distilbert(random_dataset, threshold=0.7)

fine_tune_gpt2([entry['definition'] for entry in filtered_urban_dataset], epochs=5, batch_size=4, save_path='fine_tuned_model.pth')
>>>>>>> 79011dcd19fb8f56e6f3d54873b363bed4900767
