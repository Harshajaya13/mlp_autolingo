import torch
import torch.nn.functional as F

def generate_completion(model, tokenizer, config, prompt, max_new_tokens=100):
    model.eval()
    
    token_ids = tokenizer.encode(prompt)
    
    if not token_ids:
        token_ids = [tokenizer.stoi.get(' ', 0)]
        
    context = torch.tensor([token_ids], dtype=torch.long, device=config.device) 
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_cond = context[:, -config.block_size :]
            logits, _ = model(context_cond)
            # Slicing the last token's logits for all vocabulary items
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) 
            context = torch.cat((context, next_token), dim=1)
            
    output_ids = context[0].tolist()
    return tokenizer.decode(output_ids)
