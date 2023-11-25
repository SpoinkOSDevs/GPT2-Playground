import requests
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gzip
from io import BytesIO

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Move the GPT-2 model to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize GPT-2 model optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Function to fetch a small portion of the Common Crawl dataset from a specific WARC file
def fetch_common_crawl_data(warc_url, num_documents=10):
    data = []

    with requests.get(warc_url, stream=True) as response:
        for record in ArchiveIterator(response.raw):
            if record.rec_type == 'response':
                try:
                    content_type = record.http_headers.get('Content-Type', '')

                    if 'text/html' in content_type:
                        html_content = record.content_stream().read().decode('utf-8', 'ignore')
                    elif 'application/gzip' in content_type:
                        compressed_data = record.content_stream().read()
                        with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as f:
                            html_content = f.read().decode('utf-8', 'ignore')
                    else:
                        # Handle other cases or skip if content type is not supported
                        print(f"Skipping record with unsupported content type: {content_type}")
                        continue

                    # Extract meaningful text data from the HTML content
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text_content = extract_text_from_html(soup)
                    if text_content:
                        data.append(text_content)
                        num_documents -= 1
                except Exception as e:
                    print(f"Error processing record: {e}")

    return data

# Function to process HTML content and extract meaningful text
def extract_text_from_html(soup):
    # Replace this with your logic to extract meaningful text from HTML
    # Here, we're simply getting all the text inside paragraph tags
    paragraphs = soup.find_all('p')
    text_content = ' '.join([paragraph.get_text() for paragraph in paragraphs])
    return text_content

# Function to process the Common Crawl data
def process_common_crawl_data(data):
    # Placeholder function, replace with actual data processing logic
    # For simplicity, let's assume each element in data is a text sample
    processed_data = []
    for text_sample in data:
        # Replace this with your logic for processing each text sample
        # Here, we're adding a placeholder transformation
        processed_text = process_text_sample(text_sample)
        processed_data.append(processed_text)
    return processed_data

# Placeholder function for processing each text sample
def process_text_sample(text_sample):
    # Replace this with your actual logic for processing each text sample
    # Here, we're returning the input text without any modification
    return text_sample

# Function to fine-tune the GPT-2 model on the Common Crawl dataset with batch training and progress bar
def fine_tune_gpt2(epochs=5, batch_size=10):
    # Replace this with the actual URL of the desired Common Crawl WARC file
    common_crawl_warc_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"
    common_crawl_data = fetch_common_crawl_data(common_crawl_warc_url, num_documents=10)

    if common_crawl_data:
        # Process Common Crawl data
        processed_data = process_common_crawl_data(common_crawl_data)

        for epoch in range(epochs):
            # Train the model on the entire dataset in batches with a progress bar
            progress_bar = tqdm(processed_data, desc=f'Epoch: {epoch + 1}/{epochs}')
            for text_sample in progress_bar:
                # Tokenize the text sample and move to device
                input_ids = tokenizer(text_sample, return_tensors="pt", truncation=True, padding=True)['input_ids'].to(device)

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

        # Save the fine-tuned model and tokenizer separately
        save_model(model, tokenizer)
    else:
        print("Error: No data found from the provided WARC file.")

# Main function
def main():
    fine_tune_gpt2()

if __name__ == "__main__":
    main()
