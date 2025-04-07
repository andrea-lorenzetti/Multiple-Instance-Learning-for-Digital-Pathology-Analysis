import torch
from models.metrics import ClassificationMetric

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metric = ClassificationMetric()

    for batch_idx, (data, labels) in enumerate(dataloader):
        data = [d.to(device) for d in data]
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metric.forward(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    accuracy, precision, recall, f1 = metric.epoch_metrics()
    return avg_loss, accuracy, precision, recall, f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metric = ClassificationMetric()
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = [d.to(device) for d in data]
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()
            metric.forward(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    accuracy, precision, recall, f1 = metric.epoch_metrics()
    return avg_loss, accuracy, precision, recall, f1
