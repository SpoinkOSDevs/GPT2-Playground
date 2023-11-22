from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load the fine-tuned GPT-2 model and tokenizer
gpt_model_path = 'fine_tuned_model/model_state_dict.pth'
gpt_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_model.load_state_dict(torch.load(gpt_model_path, map_location=gpt_device))
gpt_model.to(gpt_device)
gpt_model.eval()

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the DistilBERT model and tokenizer for filtering
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')
    generated_text = generate_text(prompt)
    return render_template('index.html', prompt=prompt, generated_text=generated_text)

def generate_text(prompt, max_length=100):
    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    input_ids = input_ids.to(gpt_device)

    attention_mask = torch.ones(input_ids.shape, device=gpt_device)

    # Adjust the temperature to 0.3 for more focused responses
    output = gpt_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.8,
        do_sample=True,
        temperature=0.3,  # Adjust temperature for more focused responses
    )

    generated_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)

    # Use DistilBERT to filter the generated text
    if not is_pg_content(generated_text):
        generated_text = "Sorry, I couldn't generate a PG response. Try again with a different prompt."

    return generated_text

def is_pg_content(text):
    inputs = distilbert_tokenizer(text, return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Assume label 1 is for PG content

    outputs = distilbert_model(**inputs, labels=labels)
    logits = outputs.logits

    # Check if any element in the tensor is above the threshold
    threshold = 0.5
    prediction = torch.sigmoid(logits)
    return torch.any(prediction > threshold).item()

if __name__ == '__main__':
    # Run the app with host and port specified
    app.run(host='0.0.0.0', port=8080, debug=False)
