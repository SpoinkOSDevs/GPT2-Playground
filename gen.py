from flask import Flask, render_template_string, request, flash, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_model/fine_tuned_model.pth'  # Update with your actual fine-tuned model path
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
    max_length = StringField('Max Length (default: 1000)')  # Increase for longer text
    temperature = StringField('Temperature (default: 0.8)')  # Adjust for creativity
    beam_size = StringField('Beam Size (default: 5)')  # Increase for more diverse results
    no_repeat_ngram_size = StringField('No Repeat Ngram Size (default: 2)')  # Adjust for variety
    top_k = StringField('Top K (default: 50)')  # Adjust for diversity
    top_p = StringField('Top P (default: 0.95)')  # Adjust for diversity
    preset_options = SelectField('Preset Options', choices=[('casual', 'Casual Conversation'), ('formal', 'Formal Writing'), ('creative', 'Creative Story')])
    submit = SubmitField('Generate')

html_template = '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Text Generation</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.10.2/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>

    <style>
        body {
            background-color: #f0f0f0; /* Light Gray */
            color: #333; /* Dark Gray */
        }

        .container {
            max-width: 800px;
            background-color: #fff; /* White */
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        h1,
        h2 {
            color: #333; /* Dark Gray */
        }

        .form-group {
            margin-bottom: 20px;
        }

        .btn-primary {
            background-color: #007bff; /* Primary Blue */
            border-color: #007bff;
        }

        .btn-primary:hover {
            background-color: #0056b3; /* Darker Blue */
            border-color: #0056b3;
        }

        .lead {
            font-size: 1.2rem;
            line-height: 1.6;
        }

        .list-group-item {
            background-color: #ffc107; /* Warning Yellow */
            color: #333; /* Dark Gray */
            border-color: #ffc107;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1 class="mt-5">GPT-2 Text Generation</h1>
        <form method="POST" class="mt-4">
            {{ form.hidden_tag() }}

            <div class="form-group">
                {{ form.preset_options.label }}
                {{ form.preset_options(class="form-control", id="presetOptions", onchange="updateOptions()") }}
            </div>

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
                {{ form.beam_size.label }}
                {{ form.beam_size(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.no_repeat_ngram_size.label }}
                {{ form.no_repeat_ngram_size(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.top_k.label }}
                {{ form.top_k(class="form-control") }}
            </div>
            <div class="form-group">
                {{ form.top_p.label }}
                {{ form.top_p(class="form-control") }}
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
</body>

</html>
'''

# Add preset options
preset_options = {
    'casual': {'max_length': 500, 'temperature': 0.8, 'beam_size': 5, 'no_repeat_ngram_size': 2, 'top_k': 50, 'top_p': 0.95},
    'formal': {'max_length': 300, 'temperature': 0.5, 'beam_size': 3, 'no_repeat_ngram_size': 1, 'top_k': 30, 'top_p': 0.9},
    'creative': {'max_length': 200, 'temperature': 0.7, 'beam_size': 7, 'no_repeat_ngram_size': 3, 'top_k': 70, 'top_p': 0.98},
}

@app.route('/', methods=['GET', 'POST'])
def index():
    form = GenerationForm()
    generated_text = None

    if form.validate_on_submit():
        prompt = form.prompt.data
        max_length = int(form.max_length.data) if form.max_length.data else 1000
        temperature = float(form.temperature.data) if form.temperature.data else 0.8
        beam_size = int(form.beam_size.data) if form.beam_size.data else 5
        no_repeat_ngram_size = int(form.no_repeat_ngram_size.data) if form.no_repeat_ngram_size.data else 2
        top_k = int(form.top_k.data) if form.top_k.data else 50
        top_p = float(form.top_p.data) if form.top_p.data else 0.95

        if form.preset_options.data and form.preset_options.data in preset_options:
            preset_values = preset_options[form.preset_options.data]
            max_length = preset_values['max_length']
            temperature = preset_values['temperature']
            beam_size = preset_values.get('beam_size', 5)
            no_repeat_ngram_size = preset_values.get('no_repeat_ngram_size', 2)
            top_k = preset_values.get('top_k', 50)
            top_p = preset_values.get('top_p', 0.95)

        try:
            generated_text = generate_text(prompt, max_length, temperature, beam_size, no_repeat_ngram_size, top_k, top_p)
        except Exception as e:
            flash(f"Error generating text: {str(e)}", 'error')

    return render_template_string(html_template, form=form, generated_text=generated_text, preset_options=preset_options)

@app.route('/_update_options')
def update_options():
    selected_option = request.args.get('selected_option', 'casual')
    options = preset_options[selected_option]
    return jsonify(options)

def generate_text(prompt, max_length=1000, temperature=0.8, beam_size=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, padding=True)
    attention_mask = torch.ones(input_ids.shape, device=device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=beam_size,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    sentences = sent_tokenize(generated_text)
    sentences = [sentence.capitalize() for sentence in sentences]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    decoded_sentences = [tokenizer.convert_tokens_to_string(tokens) for tokens in tokenized_sentences]

    combined_text = ' '.join(decoded_sentences)

    return combined_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
