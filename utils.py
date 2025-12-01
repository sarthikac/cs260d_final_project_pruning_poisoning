import torch
from torch.utils.data import Subset
import random
import numpy as np

def init_worker(worker_id):
    base_seed = torch.initial_seed() 
    unique_seed = (base_seed + worker_id) % 2**32
    np.random.seed(unique_seed)
    random.seed(unique_seed)

def subset_from_indices(dataset, indices):
    return Subset(dataset, indices)

def sample_random_indices(dataset, k, seed=0):
    random.seed(seed); idxs = list(range(len(dataset))); return random.sample(idxs, k)

def compute_poison_retention(poisoned_idx_set, subset_indices):
    kept = sum([1 for i in subset_indices if i in poisoned_idx_set])
    return kept, len(poisoned_idx_set), kept / max(1, len(poisoned_idx_set))

def compute_backdoor_asr(model, test_loader, trigger_patch_size=6, target_label=0, device='cuda'):
    model.eval(); total=0; succ=0
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1,3,1,1).to(device)
    std  = torch.tensor((0.2470, 0.2435, 0.2616)).view(1,3,1,1).to(device)
    with torch.no_grad():
        for inputs, targets in test_loader:
            imgs = inputs.clone().to(device)
            imgs = imgs * std + mean
            imgs[:, :, (32-trigger_patch_size):, (32-trigger_patch_size):] = torch.tensor([1.0, 0.0, 0.0]).view(1,3,1,1).to(device)
            imgs = (imgs - mean) / std
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            total += len(preds)
            succ += (preds == target_label).sum().item()
    return succ / total

def compute_labelflip_asr(model, test_loader, source_label=0, target_label=1, device='cuda'):
    model.eval(); total=0; succ=0
    with torch.no_grad():
        for inputs, targets in test_loader:
            mask = (targets == source_label)
            if mask.sum() == 0: continue
            batch = inputs[mask].to(device)
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            total += len(preds)
            succ += (preds == target_label).sum().item()
    return succ / total if total>0 else 0.0