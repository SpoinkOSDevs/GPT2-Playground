from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned BrokeGPT model and tokenizer
model_path = 'fine_tuned_model/model_state_dict.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a GPT2Config to match the BrokeGPT model
config = GPT2Config.from_pretrained('gpt2', num_hidden_layers=20)
model = GPT2LMHeadModel(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Memory to store user input and conversation history
user_memory = {'prompts': [], 'conversation': []}

@app.route('/')
def index():
    # Display the conversation history to the user with labels
    labeled_conversation = []
    for i, prompt in enumerate(user_memory['prompts']):
        labeled_conversation.append(f'You: {prompt}')
        labeled_conversation.append(f'BrokeGPT: {user_memory["conversation"][i]}')

    return render_template('index.html', conversation_history=labeled_conversation)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')

    # Store user input in memory
    user_memory['prompts'].append(prompt)

    # Build conversation history
    conversation_history = "\n".join(user_memory['conversation'] + user_memory['prompts'])

    generated_text = generate_text(prompt)

    # Store generated text in memory for future reference
    user_memory['conversation'].append(generated_text)

    return render_template('index.html', prompt=prompt, generated_text=generated_text, conversation_history=conversation_history)

def generate_text(prompt, max_length=100):
    input_text = f'You: {prompt}\nBrokeGPT:'
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    input_ids = input_ids.to(device)

    attention_mask = torch.ones(input_ids.shape, device=device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.1,
        do_sample=True,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
