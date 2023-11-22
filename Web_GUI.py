from flask import Flask, render_template_string, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_model/model_state_dict.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Text Generation</title>
</head>
<body>
    <h1>GPT-2 Text Generation</h1>
    <form method="POST" action="/generate">
        <label for="prompt">Enter Prompt:</label>
        <input type="text" id="prompt" name="prompt" required>
        <button type="submit">Generate</button>
    </form>
    <h2>Generated Text:</h2>
    <p>{{ generated_text }}</p>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')
    generated_text = generate_text(prompt)
    return render_template_string(html_template, prompt=prompt, generated_text=generated_text)

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    input_ids = input_ids.to(device)

    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    # Run the app with host and port specified
    app.run(host='0.0.0.0', port=8080, debug=False)
