import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def compute_forgetting_scores(model_fn, dataset, epochs=5, device='cuda', num_workers=0):
    """
    Compute forgetting scores as in Toneva et al. 2018
    Fully vectorized on GPU.
    """
    model = model_fn(device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_examples = len(dataset)
    
    correctness_history = torch.zeros((n_examples, epochs), dtype=torch.bool, device=device)
    
    fixed_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    
    for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_workers)
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f'Forget Train {epoch+1}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            start_idx = 0
            for inputs, targets in fixed_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
                correct_batch = (preds == targets) 
                
                end_idx = start_idx + len(correct_batch)
                correctness_history[start_idx:end_idx, epoch] = correct_batch
                start_idx = end_idx
    
    # A "forget" is when accuracy goes from 1 (True) -> 0 (False)    
    prev_acc = correctness_history[:, :-1] 
    curr_acc = correctness_history[:, 1:]  

    forgetting_events = (prev_acc == True) & (curr_acc == False)    
    forgetting_scores = forgetting_events.long().sum(dim=1)
    
    return forgetting_scores.cpu().numpy()