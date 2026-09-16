# Autolingo: GRU Next-Word Prediction & Autocomplete Engine

This project implements a **Gated Recurrent Unit (GRU)** Language Model in PyTorch for Next-Word Prediction and Autocomplete Text Generation.

---

## 📁 Project Structure

* **`model.py`**: PyTorch GRU Neural Network architecture (`GRUModel` / `GPT`).
* **`dataset.py`**: Memory-mapped binary dataloader (`DataLoader`).
* **`train.py`**: Model training loop with loss evaluation and checkpoint saving (`best_checkpoint.pt`).
* **`generate.py`**: Interactive text autocomplete generation script using Temperature Scaling & Top-K Sampling.
* **`prepare.py`**: Tokenization script converting text datasets into binary tokens via `tiktoken`.
* **`canary_dataset.txt`**: Source text dataset.
* **`train.bin` / `val.bin`**: Tokenized training and validation binary datasets.
* **`requirements.txt`**: Required Python packages.

---

## 💻 Setup Guide for Windows

### Step 1: Open Terminal
Open **Command Prompt (cmd)** or **PowerShell** in the project folder directory.

### Step 2: Create a Python Virtual Environment (`venv`)
Run the following command to create a virtual environment named `venv`:

```cmd
python -m venv venv
```

### Step 3: Activate the Virtual Environment

* **In Command Prompt (cmd)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

* **In PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell shows a policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

* **In Git Bash**:
  ```bash
  source venv/Scripts/activate
  ```

*(Once activated, you will see `(venv)` prefix in your terminal prompt).*

### Step 4: Install Dependencies
Run the following command to install required libraries (`torch`, `tiktoken`, `numpy`):

```cmd
pip install -r requirements.txt
```

*(Or manually: `pip install torch tiktoken numpy`)*

---

## 🚀 How to Run the Project

Since `train.bin` and `val.bin` are already included, you can start training directly!

### Step 1: Train the GRU Model
Run the training script:

```cmd
python train.py
```

* **What happens**:
  - The script trains a 2-layer GRU model on `train.bin`.
  - Every 250 steps, it evaluates loss on `val.bin`.
  - Saves the best performing model checkpoint to **`best_checkpoint.pt`**.

### Step 2: Test Text Autocomplete / Generation
Run the text generation script:

```cmd
python generate.py
```

* **What happens**:
  - Prompts you to enter text (e.g. `enter the prompt: Natural language processing`).
  - Uses Temperature Scaling ($T=0.8$) and Top-K ($K=40$) sampling to generate 50 continuation tokens.
  - Displays the completed text output in your terminal.

---

## 🛠️ Requirements Summary
- Python 3.9+
- PyTorch 2.0+
- tiktoken
- numpy
