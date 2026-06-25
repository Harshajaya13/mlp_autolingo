import torch
class Config:
    vocab_size=65
    block_size=32 
    batch_size = 16
    n_embd = 64
    lr = 1e-3
    max_steps = 2000
    eval_interval = 100
    eval_iters = 50
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __repr__(self):
        return (f"Config(n_embd={self.n_embd}, block_size={self.block_size}, "
                f"lr={self.learning_rate}, device={self.device})")


