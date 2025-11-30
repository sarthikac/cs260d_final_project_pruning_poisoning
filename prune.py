import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import copy
from train_utils import train_one_epoch


def get_prunable_layers(model):
    """Returns list of (name, module) for valid prunable layers (Conv2d, Linear)."""
    valid_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            valid_layers.append((name, m))
    return valid_layers

def prune_by_percentile(model, amount, device='cuda'):
    mask = {}
    valid_layers = get_prunable_layers(model)

    for name, m in valid_layers:
        weights = m.weight.data.abs().view(-1)        
        surviving_weights = weights[weights > 1e-9] 
        
        if len(surviving_weights) == 0:
            threshold = 0 
        else:
            threshold = torch.quantile(surviving_weights, amount)

        mask[name + '.weight'] = (m.weight.data.abs() > threshold).float()

    return mask

def apply_mask(model, mask):
    """Applies mask to model weights."""
    for name, p in model.named_parameters():
        if name in mask:
            p.data.mul_(mask[name])

def iterative_magnitude_prune_and_retrain(model_fn, train_dataset, test_loader,
                                        fraction_to_prune=0.2, iterations=2,
                                        rewind_epoch=1, epochs_per_cycle=3,
                                        batch_size=256,
                                        num_workers = 0,
                                        device='cuda'):
    model = model_fn(device=device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()

    # 1. Train to rewind point
    print(f"Training to rewind epoch {rewind_epoch}...")
    for e in range(rewind_epoch):
        train_one_epoch(model, DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), opt, crit, device=device)

    rewind_state = copy.deepcopy(model.state_dict())
    mask = None

    for it in range(iterations):
        print(f"Pruning iteration {it+1}/{iterations}...")

        mask = prune_by_percentile(model, fraction_to_prune, device=device)

        # Rewind weights, re-initialize training optimizer and scheduler for
        # re-training.
        model.load_state_dict(rewind_state)
        apply_mask(model, mask)

        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=epochs_per_cycle//2, gamma=0.1)

        # Retrain with mask enforcement
        # Create DataLoader once outside epoch loop for efficiency
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for e in range(epochs_per_cycle):
            model.train()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                opt.zero_grad()
                outputs = model(inputs)
                loss = crit(outputs, targets)
                loss.backward()

                for name, p in model.named_parameters():
                    if name in mask:
                        p.grad.mul_(mask[name])

                opt.step()
                
            apply_mask(model, mask)
            scheduler.step()

    return model, mask