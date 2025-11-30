import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def compute_el2n_scores(model_fn, dataset, epochs=5, device='cuda', num_workers = 0):
    """
    Compute EL2N scores as in Data Diet paper: average error L2 norm in early training
    """
    model = model_fn(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Store scores for each example at each epoch
    epoch_scores = [[] for _ in range(len(dataset))]
    
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    
    for epoch in range(epochs):
        print(f'EL2N Training Epoch {epoch+1}/{epochs}')
        model.train()
        
        # Training phase
        for inputs, targets in tqdm(loader, desc=f'EL2N Train {epoch+1}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation phase: compute errors for all examples
        model.eval()
        with torch.no_grad():
            for inputs, targets in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Convert to probabilities and compute error L2 norm
                probabilities = F.softmax(outputs, dim=1)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                errors = probabilities - one_hot_targets
                el2n_batch = torch.norm(errors, p=2, dim=1)
                
                # Store scores
                for i, score in enumerate(el2n_batch):
                    idx = i + len(epoch_scores) - len(el2n_batch)  # Adjust indexing
                    if idx < len(epoch_scores):
                        epoch_scores[idx].append(score.item())
    
    # Average over epochs (excluding first epoch as in paper)
    final_scores = np.zeros(len(dataset))
    for i, scores in enumerate(epoch_scores):
        if len(scores) > 1:  # Use later epochs only
            final_scores[i] = np.mean(scores[1:])
        elif len(scores) == 1:
            final_scores[i] = scores[0]
    
    return final_scores