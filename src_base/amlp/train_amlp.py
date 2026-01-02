import sys
import os
import re
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

# Running the external python file run-classifier.py from the official repo for aMLP
def run_official_classifier(
    amlp_repo: Path,
    dataset_train: Path,
    dataset_dev: Path,
    base_model_dir: Path,
    log_dir: Path,
    run_name: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    python_exe: str | None = None,
):

    # Make sure that the setting aligns with the environment where run_amlp is run
    if python_exe is None:
        python_exe = sys.executable

    # Create a path from the repo
    run_classifier_path = (amlp_repo / "run-classifier.py").resolve()
    if not run_classifier_path.exists():
        raise FileNotFoundError(f"run-classifier.py not found: {run_classifier_path}")

    # Since the script expects all the tok packages inside of CWD,
    # CWD should be set up as third_party/aMLP-japanese, not the actual CWD
    repo_root = amlp_repo.resolve()

    cmd = [
        python_exe,
        str(run_classifier_path),
        "--dataset", str(dataset_train.resolve()),
        "--val_dataset", str(dataset_dev.resolve()),
        "--base_model", str(base_model_dir.resolve()),
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate),
        "--log_dir", str(log_dir.resolve()),
        "--run_name", run_name,
    ]

    print("Running official aMLP classifier script:")
    print(f"CWD: {repo_root}")
    print("CMD:", " ".join(cmd))

    # subprocess.run for running the third-party script
    completed = subprocess.run(
        cmd, # run the assigned command
        cwd=str(repo_root), # set up cwd properly
        capture_output=True,
        text=True, # Records the output (stdout/stderr)
    )

    print("\n----run-classifier.py STDOUT----\n")
    print(completed.stdout)
    print("\n----run-classifier.py STDERR----\n")
    print(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"run-classifier.py failed with exit code {completed.returncode}. "
            "See stdout/stderr above."
        )

# Util: avoid proceeding with the extracted git repo broken
def ensure_git_repo(repo_dir: Path, repo_url: str, sentinel: str):
    """
    Ensure a git repo is present at repo_dir by checking sentinel file.
    If not present, clone repo_url into repo_dir.

    sentinel: a file path relative to repo_dir that must exist (e.g., "ja-swe24k.txt")
    """
    repo_dir = repo_dir.resolve()
    sentinel_path = repo_dir / sentinel

    if sentinel_path.exists():
        print(f"Repo already exists at: {repo_dir}")
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cloning {repo_url} into: {repo_dir}")
    cmd = ["git", "clone", repo_url, str(repo_dir)]
    completed = subprocess.run(cmd, capture_output=True, text=True)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to clone {repo_url}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    if not sentinel_path.exists():
        raise RuntimeError(
            f"Clone succeeded but sentinel not found: {sentinel_path}\n"
            f"Repo layout may have changed or wrong sentinel."
        )

    print("Clone successful")

def _sanitize_label(label):
    """
    Make a safe directory name from a label.
    """
    s = str(label).strip()
    # Replace OS path separators and control chars
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    # Keep reasonably safe chars
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s or "EMPTY"


def _write_dataset_dir(df, text_col, label_col, out_root):
    """
    Write dataframe into the directory structure expected by tanreinama/aMLP-japanese run-classifier.py:

    out_root/
      <class_name>/
        000000.txt
        000001.txt
        ...
    
    Each row becomes one .txt file.
    class_name is derived from df[label_col] and sanitized to be filesystem-safe.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    for i, row in df.reset_index(drop=True).iterrows():
        label = _sanitize_label(row[label_col])
        cls_dir = out_root / label
        cls_dir.mkdir(parents=True, exist_ok=True)

        text = "" if pd.isna(row[text_col]) else str(row[text_col])

        # Zero-padded file name for stability/readability
        fn = cls_dir / f"{i:06d}.txt"
        fn.write_text(text, encoding="utf-8")


def _copy_tokenizer_assets(tokenizer_dir, vocab_file, emoji_file, work_dir):
    vocab_src = (tokenizer_dir / vocab_file).resolve()
    emoji_src = (tokenizer_dir / emoji_file).resolve()

    if not vocab_src.exists():
        raise FileNotFoundError(f"vocab_file not found: {vocab_src}")
    if not emoji_src.exists():
        raise FileNotFoundError(f"emoji_file not found: {emoji_src}")

    dst_emoji = work_dir / "emoji.json"
    dst_vocab = work_dir / "vocabulary.txt"

    if not dst_emoji.exists():
        shutil.copyfile(emoji_src, dst_emoji)

    if not dst_vocab.exists():
        shutil.copyfile(vocab_src, dst_vocab)


def train_amlp(cfg, train_df, dev_df, out_dir):
    """
    Train aMLP using the OFFICIAL TensorFlow script (run-classifier.py) via subprocess.

    This function:
      1) creates dataset directories in run_dir/
      2) copies emoji.json into run_dir (required by official script)
      3) calls: python <run_classifier.py> --dataset ... --val_dataset ... --base_model ... --log_dir ...
      4) saves checkpoints under: run_dir/checkpoint/<run_name>/checkpoint-XXXX

    Requirements:
        You have the aMLP-japanese repo cloned locally (so run-classifier.py is a local file)
        You have the base model directory present (cfg["model"]["model_dir"]) containing:
            hparams.json, vocabulary.txt, saved_model (etc.)
        TensorFlow environment that can load the model


        The 'tokenizer' argument is not used here; official script builds its own encoder
        from vocabulary.txt + emoji.json (current dir).
    """

    third_party = Path("third_party")

    # aMLP-japanese
    amlp_repo = third_party / "aMLP-japanese"
    ensure_git_repo(
        repo_dir=third_party / "aMLP-japanese",
        repo_url="https://github.com/tanreinama/aMLP-japanese",
        sentinel="run-classifier.py",
    )

    # Japanese-BPEEncoder_V2
    ensure_git_repo(
        repo_dir=third_party / "Japanese-BPEEncoder_V2",
        repo_url="https://github.com/tanreinama/Japanese-BPEEncoder_V2",
        sentinel="ja-swe24k.txt",
    )
    run_classifier_path = amlp_repo / "run-classifier.py"

    out_dir = Path(out_dir)
    run_dir = out_dir / "amlp_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    text_col = cfg["data"]["text_col"]
    label_col = cfg["data"]["label_col"]

    model_dir = Path(cfg["model"]["model_dir"])  #　"./aMLP-base-ja"
    if not model_dir.exists():
        raise FileNotFoundError(f"Base model directory not found: {model_dir}")

    # Tokenizer emoji.json path
    tok_cfg = cfg["tokenizer"]
    tokenizer_dir = Path(tok_cfg["tokenizer_dir"]).resolve()
    _copy_tokenizer_assets(
        tokenizer_dir=tokenizer_dir,
        vocab_file=tok_cfg["vocab_file"],
        emoji_file=tok_cfg["emoji_file"],
        work_dir=run_dir,
    )

    # Write datasets in official format (.txt)
    train_ds_dir = run_dir / "dataset_train"
    dev_ds_dir = run_dir / "dataset_dev"
    _write_dataset_dir(train_df, text_col, label_col, train_ds_dir)
    _write_dataset_dir(dev_df, text_col, label_col, dev_ds_dir)

    # logging (The official script outputs it)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Make a safe directory name
    run_name = cfg["experiment"]["name"]
    run_name = _sanitize_label(run_name)

    batch_size = int(cfg["task"]["batch_size"])
    num_epochs = int(cfg["task"]["num_epochs"])
    learning_rate = float(cfg["task"]["learning_rate"])

    cmd = [
        sys.executable,
        str(run_classifier_path.resolve()),
        "--dataset",
        str(train_ds_dir),
        "--val_dataset",
        str(dev_ds_dir),
        "--base_model",
        str(model_dir.resolve()),
        "--batch_size",
        str(batch_size),
        "--num_epochs",
        str(num_epochs),
        "--learning_rate",
        str(learning_rate),
        "--log_dir",
        str(log_dir),
        "--run_name",
        run_name,
    ]

    print("Running official aMLP classifier script:")
    print("  CWD:", run_dir)
    print("  CMD:", " ".join(cmd))

    run_official_classifier(
        amlp_repo=amlp_repo,
        dataset_train=train_ds_dir,
        dataset_dev=dev_ds_dir,
        base_model_dir=model_dir,
        log_dir=log_dir,
        run_name=run_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    # Locate latest checkpoint (best-effort)
    ckpt_root = run_dir / "checkpoint" / run_name
    print("\nCheckpoints saved under:", ckpt_root)

    if ckpt_root.exists():
        ckpts = sorted([p for p in ckpt_root.glob("checkpoint-*") if p.is_dir()])
        if ckpts:
            print("Latest checkpoint:", ckpts[-1])
        else:
            print("No checkpoint-* directories found (maybe training ended before first save).")
    else:
        print("Checkpoint directory not found (unexpected).")

    return {
        "run_dir": run_dir,
        "train_ds_dir": train_ds_dir,
        "dev_ds_dir": dev_ds_dir,
        "log_dir": log_dir,
        "checkpoint_dir": ckpt_root,
        "run_name": run_name,
    }