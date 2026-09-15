# AutoLingo Engine ⌨️🧠

AutoLingo is a self-hosted, lightweight neural text autocomplete engine built on top of a PyTorch GRU recurrent neural network. It includes an interactive CLI mode for training/testing and a sleek, modern, Monkeytype-inspired Web UI for real-time autocomplete writing.

---

## Features

- **Deep Learning Core**: Built on a 2-layer Recurrent Neural Network (GRU) designed for character-level prediction.
- **Interactive Training & Inference**: `main.py` handles automatic dataset fetching (Tiny Shakespeare), training checkpoints, resuming/scratch training options, and interactive CLI text generation.
- **Sleek Web Interface**: A minimalist text completion playground mimicking the popular Monkeytype theme:
  - **Tab**: Accept the first predicted word.
  - **Right Arrow ($\rightarrow$)**: Accept the entire predicted sentence/phrase.
  - **Escape**: Reject the prediction (logs to AI stats).
- **Settings & Stats**:
  - 👑 **Usage Statistics**: Track your acceptance rate vs. rejection rate.
  - ⚙️ **Inference Settings**: Dynamically control how many characters/tokens the AI predicts ahead.

---

## Installation & Setup

### 1. Clone & Navigate to the Repository

```bash
git clone https://github.com/Harshajaya13/mlp_autolingo.git
cd mlp_autolingo
```

### 2. Install Dependencies

Ensure you have Python 3.8+ and PyTorch installed. You can install all requirements with:

```bash
pip install -r requirements.txt
```

*(If you don't have PyTorch installed, you can find installation instructions tailored to your system's CUDA support at [pytorch.org](https://pytorch.org/).)*

---

## Usage

### 1. Launching the Web Server (Direct/Recommended)

You can launch and interact with the AI directly via the Web UI. Simply run:

```bash
python3 web_server.py [port]
```

- **No CLI Setup Needed**: If the Tiny Shakespeare dataset (`data/train.txt`) is missing, `web_server.py` will automatically download and prepare the files.
- **Port**: By default, it runs on **`http://localhost:8080`**.
- **Accessing the UI**: Open `http://localhost:8080` in your web browser. Start typing, and the model will suggest real-time inline predictions!

---

### 2. Optional: CLI training & Interactive Mode

If you want to train the neural network further, customize checkpoints, or test completions inside your terminal:

```bash
python3 main.py
```

- When run, if an existing checkpoint is found at `checkpoints/model.pt`, you can choose to:
  1. Skip training and run inference inside the CLI.
  2. Resume training to update weights.
  3. Reset the checkpoint and train from scratch.

---

## Project Structure

```
├── checkpoints/          # Saved model weights (.pt files)
├── data/                 # Training and validation texts (.txt files)
├── engine/               # Core Python modules
│   ├── config.py         # Model hyperparameters and devices (CUDA/CPU)
│   ├── dataset.py        # PyTorch Dataset loaders
│   ├── generate.py       # Sequence generation helper functions
│   ├── model.py          # PyTorch GRU architecture (2-layer GRU)
│   ├── tokenizer.py      # Character-level tokenizer
│   └── train.py          # Training loop and loss evaluation
├── results/              # Metrics logs and evaluation logs
├── scripts/              # Helper utility scripts
├── web/                  # Web server frontend static assets
│   ├── app.js            # Frontend logic and hotkey listeners
│   ├── index.html        # Monkeytype-style autocomplete layout
│   └── style.css         # Styling, variables, and typography
├── main.py               # Main CLI orchestrator (train/test)
├── web_server.py         # HTTP API/Web server for frontend interaction
└── requirements.txt      # Python package requirements
```

---

## Shortcuts in Web UI

- **`Tab`**: Accepts the first predicted word and triggers the next prediction.
- **`Right Arrow`**: Accepts the entire generated snippet.
- **`Escape`**: Dismisses/clears the current prediction.
- **`Clicking anywhere`**: Returns focus to the hidden editor textarea.
