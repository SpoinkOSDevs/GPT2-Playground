from flask import Flask, render_template_string, request, flash, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import nltk
from nltk.tokenize import sent_tokenize
import string
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Load the fine-tuned model and tokenizer
model_path = 'fine_tuned_model/fine_tuned_model.pth'  # Update with your actual fine-tuned model path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the initial model configuration
config = GPT2Config.from_pretrained('gpt2-medium')
config.hidden_size = 1024  # Set the hidden size to match your fine-tuned model
model = GPT2LMHeadModel(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Get the layers from the model
layers = list(model.transformer.children())

# Function to get a new layer dynamically
def get_new_layer():
    # Get the next layer from the list (cycling through)
    next_layer = layers.pop(0)
    layers.append(next_layer)
    return next_layer

# Update the architecture of your fine-tuned model's layer
def change_layer():
    global layer  # Assuming 'layer' is a global variable

    while True:
        time.sleep(1)  # Change the layer every second (adjust as needed)
        
        # Assume 'new_layer' is the new layer you want to load
        new_layer = get_new_layer()  # Implement a function to get the new layer
        layer.load_state_dict({
            'weight': new_layer.weight[:, :layer.weight.shape[1]],
            'bias': new_layer.bias,
        })

# Start the thread to change the layer
layer_change_thread = threading.Thread(target=change_layer)
layer_change_thread.daemon = True
layer_change_thread.start()

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# Define the Flask form
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

# HTML template for rendering the form and generated text
html_template = 'templates/index.html'

# Flask route to handle the web interface
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

    return render_template_string(html_template, form=form, generated_text=generated_text)

# Flask route to handle AJAX request for updating options dynamically
@app.route('/_update_options')
def update_options():
    selected_option = request.args.get('selected_option', 'casual')
    options = preset_options[selected_option]
    return jsonify(options)

# Function to generate text based on user input
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
        temperature=temperature,
        do_sample=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    sentences = sent_tokenize(generated_text)
    sentences = [sentence.capitalize() for sentence in sentences]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    decoded_sentences = [tokenizer.convert_tokens_to_string(tokens) for tokens in tokenized_sentences]

    combined_text = ' '.join(decoded_sentences)

    return combined_text

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
