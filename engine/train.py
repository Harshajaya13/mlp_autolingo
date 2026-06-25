import torch
import torch.nn.functional as F
from engine.dataset import get_batch

@torch.no_grad()
def estimate_loss(model, train_dataset, val_dataset, config):
    out = {}
    model.eval()
    for split, dataset in [('train', train_dataset), ('val', val_dataset)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(dataset, config.batch_size, config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        
    model.train()
    return out

def train_model(model, train_dataset, val_dataset, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.train()
    
    loss_history = []
    print(f"printing training on the {config.device}")
    
    for step in range(config.max_steps):
        xb, yb = get_batch(train_dataset, config.batch_size, config.device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, train_dataset, val_dataset, config)
            print(f"Step {step:05d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            loss_history.append((step, losses['train'], losses['val']))

    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("Training finished! Saved checkpoint to checkpoints/model.pt.")
    return loss_history
