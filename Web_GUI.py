from flask import Flask, render_template_string, request, flash
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_model/model_state_dict.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a new pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Download the nltk punkt tokenizer data
nltk.download('punkt')

class GenerationForm(FlaskForm):
    prompt = StringField('Enter Prompt:')
    max_length = StringField('Max Length (default: 100)')
    temperature = StringField('Temperature (default: 1.0)')
    submit = SubmitField('Generate')

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Text Generation</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #87CEEB; /* Sky Blue */
            color: #000080; /* Navy Blue */
            font-family: 'Arial', sans-serif;
        }
        .container {
            max-width: 800px;
            background-color: #ffffff; /* White */
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1, h2 {
            color: #000080; /* Navy Blue */
        }
        .form-group {
            margin-bottom: 20px;
        }
        .btn-primary {
            background-color: #4169E1; /* Royal Blue */
            border-color: #4169E1;
        }
        .btn-primary:hover {
            background-color: #000080; /* Navy Blue */
            border-color: #000080;
        }
        .lead {
            font-size: 1.2rem;
            line-height: 1.6;
        }
        .list-group-item {
            background-color: #FFD700; /* Gold */
            color: #000080; /* Navy Blue */
            border-color: #FFD700;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mt-5">GPT-2 Text Generation</h1>
        <form method="POST" class="mt-4">
            {{ form.hidden_tag() }}
            <div class="form-group">
                {{ form.prompt.label }}
                {{ form.prompt(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.max_length.label }}
                {{ form.max_length(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.temperature.label }}
                {{ form.temperature(class="form-control") }}
            </div>
            {{ form.submit(class="btn btn-primary") }}
        </form>
        {% if generated_text %}
            <h2 class="mt-4">Generated Text:</h2>
            <p class="lead">{{ generated_text|safe }}</p>
        {% endif %}
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <ul class="list-group mt-4">
                    {% for message in messages %}
                        <li class="list-group-item list-group-item-danger">{{ message }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    form = GenerationForm()
    generated_text = None

    if form.validate_on_submit():
        prompt = form.prompt.data
        max_length = int(form.max_length.data) if form.max_length.data else 100
        temperature = float(form.temperature.data) if form.temperature.data else 1.0

        try:
            generated_text = generate_text(prompt, max_length, temperature)
        except Exception as e:
            flash(f"Error generating text: {str(e)}", 'error')

    return render_template_string(html_template, form=form, generated_text=generated_text)

def generate_text(prompt, max_length=100, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, padding=True)
    attention_mask = torch.ones(input_ids.shape, device=device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
        do_sample=True,    
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-process to form correct sentences
    sentences = sent_tokenize(generated_text)
    sentences = [sentence.capitalize() for sentence in sentences]

    # Tokenize the sentences for better handling
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    # Decode tokenized sentences
    decoded_sentences = [tokenizer.convert_tokens_to_string(tokens) for tokens in tokenized_sentences]

    # Combine sentences to ensure coherence
    combined_text = ' '.join(decoded_sentences)

    return combined_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
