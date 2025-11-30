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
        # Keep on GPU - accumulate without .item()
        running_loss += loss.detach() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0); correct += predicted.eq(targets).sum()
    # Single GPU->CPU transfer at the end
    return running_loss.item()/total, correct.item()/total

def evaluate(model, loader, device='cuda'):
    model.eval()
    total=0; correct=0; losses=0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device); targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Keep on GPU - accumulate without .item()
            losses += loss.detach()
            _, predicted = outputs.max(1)
            total += targets.size(0); correct += predicted.eq(targets).sum()
    # Single GPU->CPU transfer at the end
    return losses.item()/total, correct.item()/total

def predict_logits(model, loader, device='cuda'):
    model.eval()
    logits_list = []; targets_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu()); targets_list.append(targets)
    return torch.cat(logits_list), torch.cat(targets_list)

def train_with_history(model, train_loader, test_loader, optimizer, scheduler,
                       criterion, epochs, device='cuda'):
    """
    Train model and track per-epoch loss/accuracy history.

    Returns:
        model: trained model
        history: dict with keys 'train_losses', 'train_accs', 'test_losses', 'test_accs'
    """
    history = {
        'train_losses': [],
        'train_accs': [],
        'test_losses': [],
        'test_accs': []
    }

    for e in range(epochs):
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            desc=f'Epoch {e+1}/{epochs}', device=device
        )
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)

        # Evaluation
        test_loss, test_acc = evaluate(model, test_loader, device=device)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)

        # Step scheduler
        scheduler.step()

    return model, history