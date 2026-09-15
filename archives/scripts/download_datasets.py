import os
import urllib.request

os.makedirs('data', exist_ok=True)

# URLs
wiki_train_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
wiki_val_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt"
alice_url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"

print("Downloading WikiText-2 Train (Wikipedia)...")
wiki_train = urllib.request.urlopen(wiki_train_url).read().decode('utf-8')

print("Downloading WikiText-2 Val (Wikipedia)...")
wiki_val = urllib.request.urlopen(wiki_val_url).read().decode('utf-8')

print("Downloading Alice in Wonderland (Gutenberg)...")
alice_full = urllib.request.urlopen(alice_url).read().decode('utf-8')

# Clean up Gutenberg header/footer
try:
    alice_content = alice_full.split("*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***")[1]
    alice_content = alice_content.split("*** END OF THE PROJECT GUTENBERG EBOOK")[0]
except:
    alice_content = alice_full

# Split Alice
alice_lines = alice_content.split('\n')
split_idx = int(len(alice_lines) * 0.9)
alice_train = '\n'.join(alice_lines[:split_idx])
alice_val = '\n'.join(alice_lines[split_idx:])

# Combine
train_text = wiki_train + "\n" + alice_train
val_text = wiki_val + "\n" + alice_val

with open('data/train.txt', 'w', encoding='utf-8') as f:
    f.write(train_text)
    
with open('data/val.txt', 'w', encoding='utf-8') as f:
    f.write(val_text)

print(f"\n✅ Super-Dataset ready!")
print(f"Train size: {len(train_text)//1024} KB")
print(f"Val size: {len(val_text)//1024} KB")
