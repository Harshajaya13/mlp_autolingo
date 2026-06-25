import os
import urllib.request
import torch
from engine.config import Config
from engine.tokenizer import CharTokenizer
from engine.dataset import TextDataset, get_batch
from engine.model import TextPredictor
from engine.train import train_model
from engine.generate import generate_completion


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    config = Config()
    
    train_path = "data/train.txt"
    val_path = "data/val.txt"
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("Data files train.txt and val.txt not found in data/.")
        print("Fetching Tiny Shakespeare dataset automatically to train...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            
            n = len(text)
            train_data = text[:int(n * 0.9)]
            val_data = text[int(n * 0.9):]
            
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(train_data)
            with open(val_path, "w", encoding="utf-8") as f:
                f.write(val_data)
            print("Successfully downloaded and prepared training/validation text files!")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return
        
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokenizer = CharTokenizer(text)
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary loaded. Vocab size: {config.vocab_size}")
    
    model = TextPredictor(config).to(config.device)
    
    # Check if a trained checkpoint exists to save training time
    checkpoint_path = "checkpoints/model.pt"
    skip_training = False
    if os.path.exists(checkpoint_path):
        ans = input("Found a trained model checkpoint. Load it and skip training? (y/n) [default: y]: ").strip().lower()
        if ans in ('', 'y', 'yes'):
            print(f"Loading checkpoint from {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
            skip_training = True
            
    if not skip_training:
        print("Preparing dataset for training...")
        with open(train_path, "r", encoding="utf-8") as f:
            train_tokens = tokenizer.encode(f.read())
        with open(val_path, "r", encoding="utf-8") as f:
            val_tokens = tokenizer.encode(f.read())
            
        train_dataset = TextDataset(train_tokens, config.block_size)
        val_dataset = TextDataset(val_tokens, config.block_size)
        
        print(model)
        history = train_model(model, train_dataset, val_dataset, config)
        
        with open("results/metrics_log.csv", "w") as f:
            f.write("step,train_loss,val_loss\n")
            for step, t_loss, v_loss in history:
                f.write(f"{step},{t_loss},{v_loss}\n")

    print("\n==================================================")
    print("AutoLingo Interactive Mode")
    print("Type your prompt and press Enter to predict completions.")
    print("Type 'exit' to quit interactive mode.")
    print("==================================================")
    
    while True:
        try:
            prompt = input("\nEnter prompt: ")
            if prompt.strip().lower() == "exit":
                print("Exiting interactive mode. Goodbye!")
                break
            if not prompt:
                continue
            
            # Predict the completion
            completion = generate_completion(model, tokenizer, config, prompt, max_new_tokens=100)
            print(f"\nPrediction:\n{completion}")
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nExiting interactive mode. Goodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
