import torch
import torch.nn.functional as F
import tiktoken
from model import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = 50257
seq_len = 128
num_dims = 256
num_heads = 4
num_layers = 2

model = GPT(vocab_size, seq_len, num_dims, num_heads, num_layers, p=0.0).to(device)

checkpoint = torch.load("best_checkpoint.pt", map_location=device, weights_only=True)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()

enc = tiktoken.get_encoding("gpt2")
prompt = input("enter the prompt: ")
prompt_tokens = enc.encode(prompt)
idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

max_new_tokens = 50
temperature = 0.8
top_k = 40

print(f"\nPrompt: {prompt}")
print("--- Generating ---")

with torch.no_grad():
    for _ in range(max_new_tokens):

        idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]

        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]
        
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

generated_tokens = idx[0].tolist()
output_text = enc.decode(generated_tokens)
print(output_text)