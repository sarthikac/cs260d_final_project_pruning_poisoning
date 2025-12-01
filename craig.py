import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import heapq
import os
import time
from utils import init_worker

def craig_lazy_greedy_heap(grad_embeddings, k, num_classes=10):
    """
    CRAIG gradient matching selection using lazy evaluation with heap.

    Args:
        grad_embeddings: [n, d] array of gradient embeddings
        k: number of points to select
        num_classes: number of classes (kept for compatibility)

    Returns:
        selected: list of k selected indices
    """
    n = grad_embeddings.shape[0]

    # Normalize embeddings
    grad_norms = np.linalg.norm(grad_embeddings, axis=1, keepdims=True)
    normalized_grads = grad_embeddings / (grad_norms + 1e-8)

    # Compute full gradient sum (target for approximation)
    full_gradient_sum = normalized_grads.sum(axis=0)
    full_gradient_norm = np.linalg.norm(full_gradient_sum)

    selected = []
    current_sum = np.zeros_like(full_gradient_sum)  # Sum of selected gradients

    # Initialize heap with upper bounds on marginal gains
    # Upper bound: Cauchy-Schwarz inequality
    # dot(a, b) <= ||a|| * ||b||
    # Marginal gain = dot(current_sum + grad[idx], full_sum) - dot(current_sum, full_sum)
    #               = dot(grad[idx], full_sum)
    #               <= ||grad[idx]|| * ||full_sum||
    heap = []
    grad_norms_flat = np.linalg.norm(normalized_grads, axis=1)
    for idx in range(n):
        # Upper bound on gain using Cauchy-Schwarz
        upper_bound = grad_norms_flat[idx] * full_gradient_norm
        heapq.heappush(heap, (-upper_bound, idx, idx))

    while len(selected) < k and heap:
        neg_gain, _, idx = heapq.heappop(heap)

        if idx in selected:
            continue

        # RE-EVALUATE: compute true gradient matching gain
        candidate_grad = normalized_grads[idx]
        new_sum = current_sum + candidate_grad

        # True gain = improvement in alignment with full gradient
        true_gain = np.dot(new_sum, full_gradient_sum) - np.dot(current_sum, full_gradient_sum)

        # Check if this is truly the best candidate
        if not heap or -neg_gain <= true_gain + 1e-9:
            # Accept this candidate
            selected.append(idx)
            current_sum = new_sum
        else:
            # Re-insert with updated (true) gain
            heapq.heappush(heap, (-true_gain, idx, idx))

    return selected


def craig_greedy_gradient_matching(grad_embeddings, k, num_classes=10):
    """
    CRAIG greedy selection via gradient matching as in official implementation.
    Selects points whose gradients best approximate the full dataset's gradient sum.

    Args:
        grad_embeddings: [n, d] array of gradient embeddings
        k: number of points to select
        num_classes: number of classes (kept for compatibility)

    Returns:
        selected: list of k selected indices
    """
    n = grad_embeddings.shape[0]
    selected = []
    remaining = set(range(n))

    # Normalize gradients
    grad_norms = np.linalg.norm(grad_embeddings, axis=1, keepdims=True)
    normalized_grads = grad_embeddings / (grad_norms + 1e-8)

    # Compute full gradient sum (target for approximation)
    full_gradient_sum = normalized_grads.sum(axis=0)

    # Track current sum of selected gradients
    current_sum = np.zeros_like(full_gradient_sum)

    for _ in range(k):
        best_gain = -np.inf
        best_idx = -1

        for idx in remaining:
            # Compute marginal gain: how much better does adding idx make our approximation?
            candidate_grad = normalized_grads[idx]
            new_sum = current_sum + candidate_grad

            # Gain = improvement in alignment with full gradient
            gain = np.dot(new_sum, full_gradient_sum) - np.dot(current_sum, full_gradient_sum)

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx != -1:
            selected.append(best_idx)
            remaining.remove(best_idx)
            current_sum += normalized_grads[best_idx]

    return selected

def get_craig_grad_embeddings(model, dataset, device='cuda', num_workers=0, cache_dir=None):
    """
    Compute gradient embeddings for CRAIG as in official implementation.
    Optionally caches results to disk for faster re-runs.

    Args:
        model: PyTorch model with .features and .classifier attributes
        dataset: PyTorch dataset
        device: device to use for computation
        num_workers: number of dataloader workers
        cache_dir: optional directory to cache gradient embeddings

    Returns:
        grad_embeddings: [n, d] numpy array of gradient embeddings
    """
    # Check cache if provided
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f"grad_embeddings_{len(dataset)}.npy")
        if os.path.exists(cache_file):
            print(f"Loading cached gradient embeddings from {cache_file}")
            return np.load(cache_file)

    model.eval()
    embeddings = []

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers, worker_init_fn=init_worker)

    start_time = time.time()
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

            # Vectorized gradient computation using einsum (OPTIMIZED)
            # probs - one_hot_targets: [batch_size, num_classes]
            # features_flat: [batch_size, feature_dim]
            # Result: [batch_size, num_classes, feature_dim]
            gradient_w = torch.einsum('bc,bd->bcd', probs - one_hot_targets, features_flat)

            # Flatten to [batch_size, num_classes * feature_dim]
            # Note: We only use weight gradients, not bias gradients (as in original implementation)
            batch_embeddings = gradient_w.reshape(len(inputs), -1).cpu()

            embeddings.append(batch_embeddings)

    # Concatenate all batches
    result = torch.cat(embeddings, dim=0).numpy()
    elapsed = time.time() - start_time
    print(f"Gradient embedding extraction took {elapsed:.2f} seconds")

    # Save to cache if provided
    if cache_dir is not None:
        np.save(cache_file, result)
        print(f"Saved gradient embeddings to {cache_file}")

    return result

def select_subset_craig(model_fn, full_dataset, subset_size, device='cuda', num_workers=0,
                        use_lazy_greedy=True, cache_dir=None):
    """
    CRAIG selection using gradient matching (official implementation).

    Selects a subset whose gradient sum best approximates the full dataset's gradient.

    Args:
        model_fn: function that returns a fresh model instance (accepts device parameter)
        full_dataset: full dataset to select from
        subset_size: number of samples to select (k)
        device: device to use for computation ('cuda' or 'cpu')
        num_workers: number of dataloader workers
        use_lazy_greedy: if True, use heap-based lazy greedy (faster, O(k*m*log n));
                        if False, use standard greedy (simpler, O(k*n))
        cache_dir: optional directory to cache gradient embeddings for faster re-runs

    Returns:
        selected: list of selected indices
    """
    model = model_fn(device=device)
    print('Computing CRAIG gradient embeddings...')
    grad_emb = get_craig_grad_embeddings(model, full_dataset, device, num_workers, cache_dir)

    print(f'Running CRAIG gradient matching (lazy={use_lazy_greedy})...')
    start_time = time.time()

    if use_lazy_greedy:
        selected = craig_lazy_greedy_heap(grad_emb, subset_size)
    else:
        selected = craig_greedy_gradient_matching(grad_emb, subset_size)

    elapsed = time.time() - start_time
    print(f'CRAIG gradient matching took {elapsed:.2f} seconds')
    print(f'CRAIG selected {len(selected)} items.')
    return selected
