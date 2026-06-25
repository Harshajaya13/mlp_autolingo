import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    
    def __init__(self, token_ids, block_size):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        
    def __len__(self):
        return len(self.token_ids) - self.block_size
    
    def __getitem__(self, idx):
        x = self.token_ids[idx:idx+self.block_size]
        y = self.token_ids[idx+1:idx+1+self.block_size]
        return x, y
    
def get_batch(dataset,batch_size,device):
    idx = torch.randint(len(dataset),(batch_size,))
    x_stack = torch.stack([dataset[i][0] for i in idx])
    y_stack = torch.stack([dataset[i][1] for i in idx])
    return x_stack.to(device), y_stack.to(device)


