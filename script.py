from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.load_state_dict(torch.load('fine_tuned_model/model_state_dict.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_text = generate_text(prompt)
    return render_template('index.html', prompt=prompt, generated_text=generated_text)

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    # Run the app with host and port specified
    app.run(host='0.0.0.0', port=8080, debug=False)
