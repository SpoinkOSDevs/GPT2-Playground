from flask import Flask, render_template_string, request, flash, render_template, url_for
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from flask_frozen import Freezer


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'supersecretkey'

    # Loading fine-tuned model and tokenizer
    model_path = 'fine_tuned_model/model_state_dict.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add a new pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Download the nltk punkt tokenizer data
        nltk.download('punkt')

    except Exception as e:
        # Handle the initialization error gracefully
        print(f"Error initializing model: {str(e)}")

    class GenerationForm(FlaskForm):
        prompt = StringField('Enter Prompt:')
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
                background-color: #f8f9fa; /* Light Gray */
                color: #343a40; /* Dark Gray */
                font-family: 'Arial', sans-serif;
            }
            .container {
                max-width: 800px;
                background-color: #ffffff; /* White */
                padding: 30px;
                border-radius: 15px;
                margin-top: 50px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            }
            h1, h2 {
                color: #007bff; /* Primary Blue */
            }
            .form-group, .btn-primary {
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .btn-primary {
                transition: background-color 0.3s ease;
            }
            .btn-primary:hover {
                background-color: #0056b3; /* Dark Blue */
                border-color: #0056b3;
            }
            .lead {
                color: #0062cc; /* Darker Blue */
            }
            .list-group-item {
                background-color: #f8d7da; /* Light Red */
                border-color: #f5c6cb; /* Lighter Red */
                color: #721c24; /* Dark Red */
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
                            <li class="list-group-item">{{ message }}</li>
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

    freezer = Freezer(app)

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    @app.route('/', methods=['GET', 'POST'])
    def index():
        form = GenerationForm()
        generated_text = None

        if form.validate_on_submit():
            prompt = form.prompt.data
            generated_text = generate_text(prompt)

        return render_template_string(html_template, form=form, generated_text=generated_text)

    @freezer.register_generator
    def generate():
        yield url_for('index')

    return app

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, padding=True)
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Generating text with additional parameters
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,  # You can adjust this parameter
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
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
    create_app().run(host='0.0.0.0', port=8080, debug=False)
