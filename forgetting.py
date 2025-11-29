import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import numpy as np

def compute_forgetting_scores(model_fn, dataset, epochs=5, device='cuda', num_workers=0):
    """
    Compute forgetting scores as in Toneva et al. 2018
    Track transitions between correct/incorrect predictions
    """
    model = model_fn().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_examples = len(dataset)
    # Track correctness history for each example
    correctness_history = np.zeros((n_examples, epochs), dtype=bool)
    
    # Fixed order for consistent tracking
    fixed_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    
    for epoch in range(epochs):
        print(f'Forgetting Training Epoch {epoch+1}/{epochs}')
        
        # Training phase
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
        
        # Evaluation: record correctness for this epoch
        model.eval()
        with torch.no_grad():
            start_idx = 0
            for inputs, targets in fixed_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct_batch = (preds == targets).cpu().numpy()
                
                end_idx = start_idx + len(correct_batch)
                correctness_history[start_idx:end_idx, epoch] = correct_batch
                start_idx = end_idx
    
    # Compute forgetting events (1→0 transitions)
    forgetting_scores = np.zeros(n_examples)
    for i in range(n_examples):
        correct_sequence = correctness_history[i]
        # Count transitions from correct to incorrect
        forgetting_events = 0
        for t in range(1, epochs):
            if correct_sequence[t-1] and not correct_sequence[t]:
                forgetting_events += 1
        forgetting_scores[i] = forgetting_events
    
    return forgetting_scores