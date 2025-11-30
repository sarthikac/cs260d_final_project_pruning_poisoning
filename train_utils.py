import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, desc='', device='cuda'):
    model.train()
    running_loss = 0.0; correct = 0; total = 0
    for inputs, targets in tqdm(loader, desc=desc, leave=False):
        inputs = inputs.to(device); targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0); correct += predicted.eq(targets).sum().item()
    return running_loss/total, correct/total

def evaluate(model, loader, device='cuda'):
    model.eval()
    total=0; correct=0; losses=0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device); targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0); correct += predicted.eq(targets).sum().item()
    return losses/total, correct/total

def predict_logits(model, loader, device='cuda'):
    model.eval()
    logits_list = []; targets_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu()); targets_list.append(targets)
    return torch.cat(logits_list), torch.cat(targets_list)