import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function to scrape Urban Dictionary for definitions
def scrape_urban_dictionary(term):
    url = f'https://www.urbandictionary.com/define.php?term={term}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    definitions = soup.find_all('div', class_='meaning')

    if definitions:
        return [defn.get_text() for defn in definitions]
    else:
        return None

# Function to fine-tune the GPT-2 model in real-time
def fine_tune_gpt2(terms, epochs=1):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    for epoch in range(epochs):
        for term in terms:
            definitions = scrape_urban_dictionary(term)
            
            if definitions:
                # Fine-tune the model with scraped definitions
                inputs = tokenizer(definitions, return_tensors="pt", truncation=True, padding=True)
                labels = inputs["input_ids"].clone()

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f'Term: {term}, Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}')

# List of terms to fine-tune on (replace this with your terms)
terms_to_fine_tune = ['machine learning', 'python', 'internet']

# Fine-tune the model
fine_tune_gpt2(terms_to_fine_tune)
