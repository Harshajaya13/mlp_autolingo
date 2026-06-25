import torch
class Config:
    vocab_size=65
    block_size=64 
    batch_size = 32
    n_embd = 128
    lr = 1e-3
    max_steps = 30000
    eval_interval = 250
    eval_iters = 50
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __repr__(self):
        return (f"Config(n_embd={self.n_embd}, block_size={self.block_size}, "
                f"lr={self.lr}, device={self.device})")


