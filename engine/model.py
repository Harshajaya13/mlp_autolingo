import torch
import torch.nn as nn

class TextPredictor(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size,config.n_embed)
        self.position_embedding = nn.Embedding(config.block_size,config.n_embed)
        self.linear_hidden = nn.Linear(config.n_embed,config.embed*4)
        self.linear_output = nn.Linear(config.n_embed*4,config.vocab_size)
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward_pass(self,idx,targets=None):
        B,T = idx.shape
        tok_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(torch.arange(T,device=self.cofig.device))
        
        x = tok_embed+pos_embed
        
        x = torch.relu(self.linear_hidden(x))
        logits = self.linear_output(x)
        
        loss=None
        if targets is not None:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T,C)
            targets_flat = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
            
        return logits,loss

    
    def __repr__(self):
        return f"TextPredictor(vocab={self.config.vocab_size}, params={self.count_parameters()})"

