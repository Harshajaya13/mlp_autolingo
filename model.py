import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    def __init__(self, vocab_size, seq_len=128, num_dims=256, num_heads=None, num_layers=2, p=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, num_dims)
        self.drop = nn.Dropout(p)
        self.gru = nn.GRU(
            input_size=num_dims,
            hidden_size=num_dims,
            num_layers=num_layers,
            batch_first=True,
            dropout=p if num_layers > 1 else 0.0,
        )
        self.lm_head = nn.Linear(num_dims, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)
        x = self.drop(x)
        out, _ = self.gru(x)
        logits = self.lm_head(out)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, logits.size(-1))
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

GPT = GRUModel
GRU = GRUModel