from pathlib import Path
import torch
import random
import numpy as np

def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, step=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "torch_rng_state": torch.random.get_rng_state(),
    }

    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    try:
        ckpt["numpy_rng_state"] = np.random.get_state()
        ckpt["py_rng_state"] = random.getstate()
    except Exception:
        pass

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    torch.save(ckpt, str(path))

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", strict=True):
    path = Path(path)
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    # restore RNG (optional but nice)
    if "torch_rng_state" in ckpt:
        torch_state = ckpt["torch_rng_state"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().cpu().to(torch.uint8)
        torch.random.set_rng_state(torch_state)

    if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
        cuda_states = ckpt["cuda_rng_state_all"]
        fixed = []
        for s in cuda_states:
            if isinstance(s, torch.Tensor):
                fixed.append(s.detach().cpu().to(torch.uint8))
            else:
                fixed.append(torch.as_tensor(s, dtype=torch.uint8))
        torch.cuda.set_rng_state_all(fixed)
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "py_rng_state" in ckpt:
        random.setstate(ckpt["py_rng_state"])

    start_epoch = ckpt.get("epoch", 0) + 1
    step = ckpt.get("step", 0)
    return start_epoch, step

def inspect_checkpoint(ckpt_path="checkpoints/baseline.pt", map_location="cpu"):
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[inspect_checkpoint] Not found: {p}")
        return None

    size = p.stat().st_size
    ckpt = torch.load(str(p), map_location=map_location, weights_only=False)

    print(f"[inspect_checkpoint] Found: {p} (bytes={size})")
    print(f"[inspect_checkpoint] Keys: {list(ckpt.keys())}")
    print(f"[inspect_checkpoint] epoch={ckpt.get('epoch')}, step={ckpt.get('step')}")

    return ckpt