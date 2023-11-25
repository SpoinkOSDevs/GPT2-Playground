import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import questionary
from bs4 import BeautifulSoup
from cc_webgraph import Crawler

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Move the GPT-2 model to the appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize GPT-2 model optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Function to fetch the Common Crawl dataset
def fetch_common_crawl_dataset(url, num_pages=1):
    # Crawl Common Crawl pages
    crawler = Crawler()
    data = []
    
    for page in tqdm(crawler.iter(url, num_pages=num_pages), desc='Fetching Common Crawl'):
        # Process the HTML content of each page
        soup = BeautifulSoup(page.content, 'html.parser')
        # Extract meaningful text data from the HTML content (replace this with your logic)
        text_content = extract_text_from_html(soup)
        if text_content:
            data.append(text_content)
    
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
    return [text_sample for text_sample in data if text_sample]

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

# Function to fine-tune the GPT-2 model on the Common Crawl dataset with batch training and progress bar
def fine_tune_gpt2(epochs=1, batch_size=4):
    # Fetch Common Crawl data
    common_crawl_url = "http://commoncrawl.org/2020/10"  # Replace with the actual Common Crawl URL
    num_pages = 1  # Adjust the number of pages based on your needs
    common_crawl_data = fetch_common_crawl_dataset(common_crawl_url, num_pages=num_pages)

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

# Main function
def main():
    while True:
        main_screen()

if __name__ == "__main__":
    main()
