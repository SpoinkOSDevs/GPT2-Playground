from flask import Flask, render_template_string, request, flash, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# EXTREEEEME VERBOSE: Loading fine-tuned model and tokenizer
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
    max_length = StringField('Max Length (default: 100)')
    temperature = StringField('Temperature (default: 1.0)')
    top_k = StringField('Top-K (default: 50)')
    top_p = StringField('Top-P (default: 0.95)')
    num_beams = StringField('Number of Beams (default: 5)')
    do_sample = StringField('Enable Sampling? (default: True)')
    theme = SelectField('Theme', choices=[('light', 'Light'), ('dark', 'Dark')])
    submit = SubmitField('Generate')

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Add your existing meta tags and links here -->

    <!-- Improved styling for more options -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            {% if form.theme.data == 'dark' %}
                background-color: #1e1e1e; /* Dark Gray */
                color: #ffffff; /* White */
            {% else %}
                background-color: #add8e6; /* Light Blue */
                color: #000080; /* Navy Blue */
            {% endif %}
            font-family: 'Arial', sans-serif;
        }
        .container {
            max-width: 800px;
            background-color: {% if form.theme.data == 'dark' %} #333333; {% else %} #ffffff; {% endif %};
            padding: 30px;
            border-radius: 15px;
            margin-top: 50px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        h1, h2 {
            color: {% if form.theme.data == 'dark' %} #ffffff; {% else %} #000080; {% endif %};
        }
        /* Add your existing styles here */

        /* New styles for shadows */
        .form-group, .btn-primary {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .btn-primary {
            transition: background-color 0.3s ease;
        }
        .btn-primary:hover {
            background-color: {% if form.theme.data == 'dark' %} #000080; {% else %} #4169E1; {% endif %};
            border-color: {% if form.theme.data == 'dark' %} #000080; {% else %} #4169E1; {% endif %};
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
            <div class="form-group">
                {{ form.top_k.label }}
                {{ form.top_k(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.top_p.label }}
                {{ form.top_p(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.num_beams.label }}
                {{ form.num_beams(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.do_sample.label }}
                {{ form.do_sample(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.theme.label }}
                {{ form.theme(class="form-control") }}
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

# EXTREEEEME VERBOSE: Add error handler for 500 Internal Server Error
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    form = GenerationForm()
    generated_text = None

    if form.validate_on_submit():
        prompt = form.prompt.data
        max_length = int(form.max_length.data) if form.max_length.data else 100
        temperature = float(form.temperature.data) if form.temperature.data else 1.0
        top_k = int(form.top_k.data) if form.top_k.data else 50
        top_p = float(form.top_p.data) if form.top_p.data else 0.95
        num_beams = int(form.num_beams.data) if form.num_beams.data else 5
        do_sample = form.do_sample.data.lower() == 'true' if form.do_sample.data else True

        try:
            # EXTREEEEME VERBOSE: Generating text with additional options
            generated_text = generate_text(prompt, max_length, temperature, top_k, top_p, num_beams, do_sample)
        except Exception as e:
            flash(f"Error generating text: {str(e)}", 'error')
            # EXTREEEEME VERBOSE: Raise an exception to trigger 500 error
            raise

    return render_template_string(html_template, form=form, generated_text=generated_text)

def generate_text(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, num_beams=5, do_sample=True):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, padding=True)
    attention_mask = torch.ones(input_ids.shape, device=device)

    # EXTREEEEME VERBOSE: Generating text with additional parameters
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        do_sample=do_sample,    
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

return app

if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=8080, debug=False)
