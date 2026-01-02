from pathlib import Path
import yaml
import json
import sys
import requests

# Force it to look for "config.yaml" from "Probing-JP-Politeness", NOT the current working directory
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
TOKENIZER_DIR = ROOT / "tokenizer" / "Japanese-BPEEncoder_V2"

FILES = {
    "encode_swe.py": "https://raw.githubusercontent.com/tanreinama/Japanese-BPEEncoder_V2/master/encode_swe.py",
    "emoji.json": "https://raw.githubusercontent.com/tanreinama/Japanese-BPEEncoder_V2/master/emoji.json",
    "ja-swe24k.txt": "https://raw.githubusercontent.com/tanreinama/Japanese-BPEEncoder_V2/master/ja-swe24k.txt",
}

# Download required files locally for the tokenizer if not exist
def ensure_tokenizer_files():

    # Create an upper directory (tokenizer/)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():

        # Create a path
        out = TOKENIZER_DIR / fname

        # If the file already exists, skip fetching it
        if out.exists():
            print(f"{fname} exists. Skip downloading...")
            continue
        
        print(f"{fname} downloading")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        # Write the raw version of the file
        out.write_bytes(r.content)
        print(f"{fname} saved")

    return TOKENIZER_DIR


def build_tokenizer():

    # Pull the path for the tokenizer to be downloaded
    ensure_tokenizer_files()

    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    tok_cfg = cfg["tokenizer"]

    with (TOKENIZER_DIR / tok_cfg["vocab_file"]).open(encoding="utf-8") as f:
        bpe = f.read().splitlines()

    with (TOKENIZER_DIR / tok_cfg["emoji_file"]).open(encoding="utf-8") as f:
        emoji = json.load(f)

    """
    Since Japanese-BPEEncoder_V2 is not a python package, 
    we need to let the installer to search this package for modules.
    Insert the file path into the first row of the import list (the highest priority), 
    so that Japanese-BPEEncoder_V2 can be properly imported. 
    """

    sys.path.insert(0, str(TOKENIZER_DIR))
    from encode_swe import SWEEncoder_ja

    # Load the tokenizer
    enc = SWEEncoder_ja(bpe, emoji)

    return enc

if __name__ == "__main__":
    enc = build_tokenizer()

    # Test
    # p = enc.encode("今日は日曜焼き肉定食をたべる")
    # print(p)
    # print(enc.decode(p))
    # print([enc.decode([i]) for i in p])
