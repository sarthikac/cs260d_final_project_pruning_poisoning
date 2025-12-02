import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import heapq
import os
import time
from tqdm import tqdm
from utils import init_worker
from GradualWarmupScheduler import GradualWarmupScheduler

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
                        use_lazy_greedy=True, cache_dir=None, pretrain_epochs=20, warmup_epochs=10, seed=0,
                        per_class=True, num_classes=10):
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
        pretrain_epochs: number of epochs to train before computing gradients (default: 5)
        warmup_epochs: number of warmup epochs for learning rate (default: 2)
        seed: random seed for reproducibility (default: 0)
        per_class: if True, perform selection separately per class for better balance (default: True)
        num_classes: number of classes in dataset (default: 10 for CIFAR-10)

    Returns:
        selected: list of selected indices
    """
    model = model_fn(device=device)

    # Train model for a few epochs before computing gradients (prevents random initialization bias)
    print(f'Pre-training CRAIG model for {pretrain_epochs} epochs (warmup: {warmup_epochs})...')
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Create main scheduler (after warmup)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=pretrain_epochs-warmup_epochs, eta_min=1e-5
    )

    # Create warmup scheduler that wraps the cosine scheduler
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1.0, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Create training loader with deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(full_dataset, batch_size=128, shuffle=True,
                             num_workers=num_workers, generator=g, worker_init_fn=init_worker)

    for epoch in range(pretrain_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for inputs, targets in tqdm(train_loader, desc=f'CRAIG Pretrain Epoch {epoch+1}/{pretrain_epochs}', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        print(f'  Epoch {epoch+1}/{pretrain_epochs}: Avg Loss = {epoch_loss/num_batches:.4f}')

    print('Computing CRAIG gradient embeddings on trained model...')
    grad_emb = get_craig_grad_embeddings(model, full_dataset, device, num_workers, cache_dir)

    print(f'Running CRAIG gradient matching (lazy={use_lazy_greedy}, per_class={per_class})...')
    start_time = time.time()

    if per_class:
        # Per-class selection: select proportionally from each class
        print('Performing per-class selection for better class balance...')

        # Get labels for all samples
        labels = np.array([full_dataset[i][1] for i in range(len(full_dataset))])

        # Compute per-class subset sizes (proportional to class frequency)
        selected = []
        for class_idx in range(num_classes):
            class_mask = (labels == class_idx)
            class_indices = np.where(class_mask)[0]
            class_count = len(class_indices)

            if class_count == 0:
                continue

            # Proportional allocation
            class_subset_size = int(subset_size * (class_count / len(full_dataset)))

            # Ensure at least 1 sample per class if subset_size allows
            if class_subset_size == 0 and subset_size >= num_classes:
                class_subset_size = 1

            if class_subset_size > 0:
                # Extract gradient embeddings for this class
                class_grad_emb = grad_emb[class_indices]

                # Run CRAIG selection on class subset
                if use_lazy_greedy:
                    class_selected_local = craig_lazy_greedy_heap(class_grad_emb, class_subset_size)
                else:
                    class_selected_local = craig_greedy_gradient_matching(class_grad_emb, class_subset_size)

                # Map local indices back to global indices
                class_selected_global = [class_indices[idx] for idx in class_selected_local]
                selected.extend(class_selected_global)

                print(f'  Class {class_idx}: selected {len(class_selected_global)}/{class_count} samples')

        selected = np.array(selected).tolist()
    else:
        # Global selection: select from entire dataset
        if use_lazy_greedy:
            selected = craig_lazy_greedy_heap(grad_emb, subset_size)
        else:
            selected = craig_greedy_gradient_matching(grad_emb, subset_size)

    elapsed = time.time() - start_time
    print(f'CRAIG gradient matching took {elapsed:.2f} seconds')
    print(f'CRAIG selected {len(selected)} items total.')
    return selected
