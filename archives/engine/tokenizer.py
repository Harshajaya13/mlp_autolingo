class CharTokenizer:
    
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {s: i for i, s in enumerate(self.chars)}
        self.itos = {i: s for i, s in enumerate(self.chars)}
        
    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]
    
    def decode(self, ids):
        return "".join([self.itos[id] for id in ids if id in self.itos])
    
    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"
