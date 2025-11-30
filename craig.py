import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import heapq
import os
import time

def compute_similarity_gpu(X, metric='euclidean', device='cuda'):
    """
    GPU-accelerated similarity matrix computation.

    Args:
        X: numpy array of shape [n, d] - feature vectors
        metric: 'euclidean' or 'cosine'
        device: torch device to use

    Returns:
        S: numpy array of shape [n, n] - similarity matrix
    """
    # Move to GPU
    X_tensor = torch.from_numpy(X).float().to(device)
    n = X_tensor.shape[0]

    if metric == 'cosine':
        # Normalize rows
        X_norm = X_tensor / (torch.norm(X_tensor, dim=1, keepdim=True) + 1e-8)
        # Cosine similarity via matrix multiplication
        S = torch.mm(X_norm, X_norm.T)
    elif metric == 'euclidean':
        # Pairwise euclidean: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        norms_sq = (X_tensor ** 2).sum(dim=1, keepdim=True)
        gram = torch.mm(X_tensor, X_tensor.T)
        dists_sq = norms_sq + norms_sq.T - 2 * gram
        dists_sq = torch.clamp(dists_sq, min=0)  # Numerical stability
        dists = torch.sqrt(dists_sq)

        # Convert distance to similarity
        m = torch.max(dists)
        S = m - dists
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return S.cpu().numpy()


def craig_lazy_greedy_heap(grad_embeddings, k, num_classes=10):
    """
    Lazy greedy selection with heap optimization.
    Uses heap to maintain upper bounds on marginal gains and avoids
    redundant computations.

    Args:
        grad_embeddings: [n, d] numpy array of gradient embeddings
        k: number of samples to select
        num_classes: number of classes (unused, for API compatibility)

    Returns:
        selected: list of selected indices
    """
    n = grad_embeddings.shape[0]

    # Normalize gradients
    grad_norms = np.linalg.norm(grad_embeddings, axis=1, keepdims=True)
    normalized_grads = grad_embeddings / (grad_norms + 1e-8)

    # Compute full gradient sum
    full_gradient_sum = normalized_grads.sum(axis=0)

    selected = []
    current_sum = np.zeros_like(full_gradient_sum)

    # Initialize max heap with all candidates
    # Python heapq is min-heap, so negate gains for max-heap behavior
    # Use idx as tiebreaker for deterministic ordering when gains are equal
    heap = []
    for idx in range(n):
        # Initial gain: alignment with full gradient
        gain = np.dot(normalized_grads[idx], full_gradient_sum)
        heapq.heappush(heap, (-gain, idx, idx))  # (negative gain, tiebreaker, actual_idx)

    # Greedy selection with lazy evaluation
    reeval_count = 0
    while len(selected) < k and heap:
        # Pop candidate with highest upper bound
        neg_gain, _, idx = heapq.heappop(heap)  # Unpack with tiebreaker

        if idx in selected:
            continue

        # Re-evaluate marginal gain (lazy evaluation)
        # Marginal gain = dot(current_sum + candidate, full_sum) - dot(current_sum, full_sum)
        #               = dot(candidate, full_sum)
        # This is constant for each candidate, so lazy evaluation is not needed!
        # However, we keep the structure for consistency
        true_gain = np.dot(normalized_grads[idx], full_gradient_sum)
        reeval_count += 1

        # Check if still the best (or no other candidates)
        if not heap or -neg_gain <= true_gain + 1e-9:  # Small tolerance for numerical errors
            # Add to selection
            selected.append(idx)
            current_sum += normalized_grads[idx]
        else:
            # Re-insert with updated gain and tiebreaker
            heapq.heappush(heap, (-true_gain, idx, idx))

    return selected


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
    
    # Precompute current score once
    current_score = np.dot(current_sum, full_gradient_sum)

    for _ in range(k):
        if not remaining:
            break

        # OPTIMIZED: Vectorized gain computation for all remaining candidates
        # Sort for deterministic ordering when gains are equal
        remaining_arr = np.array(sorted(remaining))
        candidate_grads = normalized_grads[remaining_arr]  # [M, D] where M = len(remaining)

        # Vectorized dot products: [M, D] @ [D] -> [M]
        candidate_scores = np.dot(candidate_grads, full_gradient_sum)
        gains = candidate_scores  # Marginal gain is just the dot product with candidate

        # Find best candidate
        best_local_idx = np.argmax(gains)
        best_idx = remaining_arr[best_local_idx]

        # Update selection
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_sum += normalized_grads[best_idx]
        current_score = np.dot(current_sum, full_gradient_sum)
    
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

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)

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
    CRAIG selection using official gradient matching approach.

    Args:
        model_fn: function that returns a fresh model instance
        full_dataset: full dataset to select from
        subset_size: number of samples to select
        device: device to use
        num_workers: dataloader workers
        use_lazy_greedy: if True, use heap-based lazy greedy (faster); if False, use vectorized greedy
        cache_dir: optional directory to cache gradient embeddings

    Returns:
        selected: list of selected indices
    """
    model = model_fn(device=device)
    print('Computing CRAIG gradient embeddings...')
    grad_emb = get_craig_grad_embeddings(model, full_dataset, device, num_workers, cache_dir)

    print(f'Running CRAIG greedy selection (lazy={use_lazy_greedy})...')
    start_time = time.time()

    if use_lazy_greedy:
        selected = craig_lazy_greedy_heap(grad_emb, subset_size)
    else:
        selected = craig_greedy_gradient_matching(grad_emb, subset_size)

    elapsed = time.time() - start_time
    print(f'CRAIG selection took {elapsed:.2f} seconds')
    print(f'CRAIG selected {len(selected)} items.')
    return selected
