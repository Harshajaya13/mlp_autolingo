"""
download_tinystories.py

Downloads the TinyStories dataset directly from Hugging Face:
  https://huggingface.co/datasets/roneneldan/TinyStories

Saves the downloaded text into 'canary_dataset.txt' so you can run:
  1. python download_tinystories.py [max_mb]
  2. python prepare.py
  3. python train.py
"""

import sys
import urllib.request

OUTPUT_FILE = "canary_dataset.txt"
# Default dataset download size limit in MB (e.g. 50 MB text)
DEFAULT_MAX_MB = 50
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"


def download_tinystories(max_mb):
    max_bytes = max_mb * 1024 * 1024 if max_mb > 0 else None
    print(f"--- TinyStories Downloader ---")
    print(f"Source URL: {DATASET_URL}")
    print(f"Output File: {OUTPUT_FILE}")
    if max_bytes:
        print(f"Target Size Limit: {max_mb} MB")
    else:
        print("Target Size Limit: Full dataset (~1.5 GB)")

    print("\nConnecting to Hugging Face...")
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})

    bytes_downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunk

    try:
        with urllib.request.urlopen(req) as response, open(OUTPUT_FILE, "wb") as f_out:
            print("Downloading text stream...")
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                if max_bytes and (bytes_downloaded + len(chunk)) > max_bytes:
                    allowed = max_bytes - bytes_downloaded
                    f_out.write(chunk[:allowed])
                    bytes_downloaded += allowed
                    break

                f_out.write(chunk)
                bytes_downloaded += len(chunk)

                current_mb = bytes_downloaded / (1024 * 1024)
                print(f" -> Downloaded {current_mb:.2f} MB...", end="\r", flush=True)

        total_mb = bytes_downloaded / (1024 * 1024)
        print(f"\n\n[SUCCESS] Saved {total_mb:.2f} MB to '{OUTPUT_FILE}'.")
        print("\nNext steps on your machine:")
        print("  1. python prepare.py   (tokenize into train.bin & val.bin)")
        print("  2. python train.py     (train your model)")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Allow passing target size in MB via CLI argument, e.g.: python download_tinystories.py 100
    target_mb = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_MB
    download_tinystories(target_mb)
