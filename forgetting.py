import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import init_worker, set_all_random_seeds, RANDOM_SEED

def compute_forgetting_scores(model_fn, dataset, epochs=5, batch_size=128, device='cuda', num_workers=0):
    """
    Compute forgetting scores as in Toneva et al. 2018.
    """
    set_all_random_seeds()
    model = model_fn(device=device, seed=RANDOM_SEED)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_examples = len(dataset)

    # Track correctness history on GPU (Speed update)
    correctness_history = torch.zeros((n_examples, epochs), dtype=torch.bool, device=device)

    # Fixed loader for consistent evaluation (Determinism update)
    fixed_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers,
                              worker_init_fn=init_worker)

    for epoch in range(epochs):
        # Create generator for deterministic shuffling (Determinism update)
        g = torch.Generator()
        g.manual_seed(RANDOM_SEED + epoch)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  generator=g, worker_init_fn=init_worker)
        
        # Training phase
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f'Forget Train {epoch+1}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluation phase: record correctness for this epoch
        model.eval()
        with torch.no_grad():
            start_idx = 0
            for inputs, targets in fixed_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
                # Vectorized comparison (Speed update)
                correct_batch = (preds == targets) 
                
                end_idx = start_idx + len(correct_batch)
                correctness_history[start_idx:end_idx, epoch] = correct_batch
                start_idx = end_idx
        
    # Get predictions at time t-1 and time t
    prev_acc = correctness_history[:, :-1]
    curr_acc = correctness_history[:, 1:]

    # A "forgetting event" is exactly when prev was True and curr is False
    forgetting_events = (prev_acc == True) & (curr_acc == False)
    
    # Sum events across epochs
    forgetting_scores = forgetting_events.long().sum(dim=1)

    return forgetting_scores.cpu().numpy()
