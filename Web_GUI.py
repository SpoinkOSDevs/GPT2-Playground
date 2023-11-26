from flask import Flask, render_template_string, request, flash, render_template, url_for
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from flask_freeze import Freezer

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
        # ... (unchanged form definition)

    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <!-- ... (unchanged HTML template) -->
    </html>
    '''

    freezer = Freezer(app)

    # Add error handler for 500 Internal Server Error
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    @app.route('/', methods=['GET', 'POST'])
    def index():
        # ... (unchanged index route)

    @freezer.register_generator
    def generate():
        yield url_for('index')

    return app

if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=8080, debug=False)
