from __future__ import annotations
from pathlib import Path
import json

def eval_amlp(cfg, tokenizer, ckpt_path, test_df, out_dir):
    run_dir = out_dir / "amlp_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Upcoming...
    # - checkpoint load
    # - test inference
    # - metrics（accuracy, f1, macro-f1??）

    test_df.to_csv(run_dir / "test.csv", index=False)

    metrics = {"accuracy": 0.0, "macro_f1": 0.0}  # TODO

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return metrics