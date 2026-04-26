import yaml
from transformers import get_scheduler
from tqdm import tqdm
from torch.optim import AdamW
import torch
from pathlib import Path
    
def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(cfg, train_dl, model, device):

    sched = cfg["scheduler"]
    sched_name = sched["name"]
    num_warmup_steps = sched["warmup_steps"]
    epochs = cfg["task"]["num_epochs"]
    lr = cfg["task"]["learning_rate"]
    num_training_steps = len(train_dl) * epochs

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    lr_scheduler = get_scheduler(
        sched_name, 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps)  

    start_epoch = 0
    global_step = 0

    for epoch in range(start_epoch, epochs):
        # set the model to training mode
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dl, desc="Train Per Batch", unit="batch"):
            batch = {k: v.to(device) for k, v in batch.items() if k != "offset_mapping"}

            # reset the gradient descent
            optimizer.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()

            # updates the parameters and the learning rate
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
        
        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")