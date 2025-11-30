import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def craig_greedy_gradient_matching(grad_embeddings, k, num_classes=10):
    """
    CRAIG greedy selection via gradient matching as in official implementation
    """
    n = grad_embeddings.shape[0]
    selected = []
    remaining = set(range(n))
    
    # Normalize gradients
    grad_norms = np.linalg.norm(grad_embeddings, axis=1, keepdims=True)
    normalized_grads = grad_embeddings / (grad_norms + 1e-8)
    
    # Compute full gradient sum (approximation)
    full_gradient_sum = normalized_grads.sum(axis=0)
    
    current_sum = np.zeros_like(full_gradient_sum)
    
    for _ in range(k):
        best_gain = -float('inf')
        best_idx = -1
        
        for idx in remaining:
            # Compute marginal gain
            candidate_grad = normalized_grads[idx]
            new_sum = current_sum + candidate_grad
            
            # Compute coverage gain (alignment with full gradient)
            gain = np.dot(new_sum, full_gradient_sum) - np.dot(current_sum, full_gradient_sum)
            
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx != -1:
            selected.append(best_idx)
            remaining.remove(best_idx)
            current_sum += normalized_grads[best_idx]
    
    return selected

def get_craig_grad_embeddings(model, dataset, device=device):
    """
    Compute gradient embeddings for CRAIG as in official implementation
    """
    model.eval()
    embeddings = []
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            features = model.features(inputs)
            features_flat = torch.flatten(features, 1)
            logits = model.classifier(features_flat)
            
            # Compute gradient w.r.t. last layer weights
            probs = F.softmax(logits, dim=1)
            one_hot_targets = F.one_hot(targets, num_classes=10).float()
            gradient_w = torch.matmul((probs - one_hot_targets).T, features_flat)
            
            # Flatten gradient matrix for each example
            batch_embeddings = []
            for i in range(len(inputs)):
                # Gradient contribution of this example
                grad_contrib = torch.outer(probs[i] - one_hot_targets[i], features_flat[i])
                batch_embeddings.append(grad_contrib.flatten().cpu())
            
            embeddings.extend(batch_embeddings)
    
    return torch.stack(embeddings).numpy()

def select_subset_craig(model_fn, full_dataset, subset_size, device='device'):
    """
    CRAIG selection using official gradient matching approach
    """
    model = model_fn().to(device)
    print('Computing CRAIG gradient embeddings...')
    grad_emb = get_craig_grad_embeddings(model, full_dataset, device)
    print('Running CRAIG greedy selection...')
    selected = craig_greedy_gradient_matching(grad_emb, subset_size)
    print(f'CRAIG selected {len(selected)} items.')
    return selected