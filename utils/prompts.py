import tiktoken

def count_tokens(text, model_name):
    """Count number of tokens in text for given model"""
    encoding = tiktoken.encoding_for_model(model_name) # Use default GPT encoding
    return len(encoding.encode(text))

# Generate and analyze prompt
def analyze_prompt(prompt_text, model_name="gpt-4o-mini"):
    num_tokens = count_tokens(prompt_text, model_name)
    print(f"Prompt Text: {prompt_text}")
    print(f"Prompt Input Tokens: {num_tokens}")
