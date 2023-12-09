tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)