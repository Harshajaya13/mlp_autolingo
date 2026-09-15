import torch
import torch.nn as nn

class TextPredictor(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.rnn = nn.GRU(config.n_embd, config.n_embd, num_layers=2, batch_first=True)
        self.linear_output = nn.Linear(config.n_embd, config.vocab_size)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        
        x, _ = self.rnn(x)
        
        logits = self.linear_output(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    def __repr__(self):
        return f"TextPredictor(RNN, vocab={self.config.vocab_size}, params={self.count_parameters()})"

