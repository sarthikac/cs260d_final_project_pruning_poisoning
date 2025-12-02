import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import init_worker

def compute_el2n_scores(model_fn, dataset, epochs=5, batch_size=128, device='cuda', num_workers=0, seed=0):
    """
    Compute EL2N scores as in Data Diet paper: average error L2 norm in early training
    """
    model = model_fn(device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Store scores on GPU as tensors (one per epoch)
    epoch_scores_gpu = []

    # Create generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                       generator=g, worker_init_fn=init_worker)

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
        epoch_scores_batch = []
        with torch.no_grad():
            eval_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers,
                                    worker_init_fn=init_worker)
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Convert to probabilities and compute error L2 norm
                probabilities = F.softmax(outputs, dim=1)
                one_hot_targets = F.one_hot(targets, num_classes=10).float()
                errors = probabilities - one_hot_targets
                el2n_batch = torch.norm(errors, p=2, dim=1)

                # Keep on GPU - accumulate batches
                epoch_scores_batch.append(el2n_batch)

        # Concatenate all batches for this epoch (still on GPU)
        epoch_scores_gpu.append(torch.cat(epoch_scores_batch))

    # Transfer to CPU once and average over epochs (excluding first epoch as in paper)
    epoch_scores_cpu = torch.stack(epoch_scores_gpu).cpu().numpy()  # Shape: (epochs, n_samples)

    if epochs > 1:
        final_scores = np.mean(epoch_scores_cpu[1:], axis=0)  # Average epochs 1..end
    else:
        final_scores = epoch_scores_cpu[0]

    return final_scores