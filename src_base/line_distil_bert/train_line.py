from torch.utils.data import Dataset, DataLoader
import yaml
from transformers import get_scheduler
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
import torch

class PolitenessDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get tokenized inputs
        item = {key: val[idx] for key, val in self.encodings.items()} 
        # Add corresponding label
        item["labels"] = self.labels[idx] 
        return item
    
def load_yaml(path): # "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_model(cfg, train_enc, dev_enc, test_enc, train_labels, dev_labels, test_labels):

    bert = cfg["model"]
    batch_size = cfg["task"]["batch_size"]
    shuffle = cfg["task"]["shuffle"]
    LineDistilBERT = bert["name"]
    num_labels = bert["num_labels"]

    # Create datasets
    train_dataset = PolitenessDataset(train_enc, train_labels)
    dev_dataset = PolitenessDataset(dev_enc, dev_labels)
    test_dataset = PolitenessDataset(test_enc, test_labels)

    # Create DataLoaders for batch training
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    model = AutoModelForSequenceClassification.from_pretrained(LineDistilBERT, num_labels=num_labels)

    print("Model successfully set up")

    return train_dataloader, dev_dataloader, test_dataloader, model

def train(cfg, train_dl, model):

    sched = cfg["scheduler"]
    sched_name = sched["name"]
    num_warmup_steps = sched["warmup_steps"]
    epochs = cfg["task"]["num_epochs"]
    lr = cfg["task"]["learning_rate"]
    num_training_steps = len(train_dl) * 3

    optimizer = AdamW(model.parameters(), lr=lr)

    lr_scheduler = get_scheduler(
        sched_name, 
        optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_training_steps)  
    
    # Define loss function (CrossEntropy for classification)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # set the model to training mode
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dl, desc="Per Batch", unit="batch"):
            batch = {key: val for key, val in batch.items()}
            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()

            # reset the gradient descent
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # updates the parameters and the learning rate
            optimizer.step()
            lr_scheduler.step()
        
        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")