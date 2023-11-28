from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sys

def generate_text(prompt, max_length=100):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.load_state_dict(torch.load('fine_tuned_model/model_state_dict.pth'))
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    # Get the prompt from command line arguments
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Default prompt"

    # Generate text and print it
    result = generate_text(prompt)
    print(result)
