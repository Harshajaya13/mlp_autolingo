import json
import os
import sys
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch

# Add workspace root to python path to ensure engine imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engine.config import Config
from engine.tokenizer import CharTokenizer
from engine.model import TextPredictor
from engine.generate import generate_completion

# Ensure target directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Global model config/instances loaded once
config = Config()
train_path = "data/train.txt"
val_path = "data/val.txt"

if not os.path.exists(train_path):
    print("Data files train.txt not found in data/.")
    print("Fetching Tiny Shakespeare dataset automatically to run the UI...")
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
        print("Successfully downloaded and prepared text files!")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)

with open(train_path, "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = CharTokenizer(text)
config.vocab_size = tokenizer.vocab_size

model = TextPredictor(config).to(config.device)
checkpoint_path = "checkpoints/model.pt"
if os.path.exists(checkpoint_path):
    print(f"Loading trained weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
else:
    print("Warning: No checkpoint found at checkpoints/model.pt. Running with uninitialized weights.")
model.eval()

class AutoLingoRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP request logging to keep terminal output clean
        pass

    def do_GET(self):
        # Route static files
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("web/index.html", "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/style.css":
            self.send_response(200)
            self.send_header("Content-Type", "text/css")
            self.end_headers()
            with open("web/style.css", "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/app.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.end_headers()
            with open("web/app.js", "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_new_tokens", 25)
                
                # Check for empty prompt
                if not prompt:
                    pred_text = ""
                else:
                    # Generate predictions
                    pred_text = generate_completion(model, tokenizer, config, prompt, max_new_tokens=max_tokens, return_new_only=True)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"completion": pred_text}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, AutoLingoRequestHandler)
    print(f"AutoLingo monkeytype-style UI is running at http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
