import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import new_backdoor_dataset
from train_utils import *
from utils import *
from model import get_model
from el2n import compute_el2n_scores
from forgetting import compute_forgetting_scores
from craig import select_subset_craig
from prune import iterative_magnitude_prune_and_retrain
from torch.utils.data import DataLoader

def run_all_methods(poison_frac=0.02, subset_frac=0.25, selection_methods=None, device='cuda', num_workers=4, seed=0,
                    data_root='./data', train_epochs=50, batch_size=128, test_loader=None,
                    el2n_epochs=20, forget_epochs=20, craig_pretrain_epochs=10,
                    fraction_to_prune=0.3, iterations=2, rewind_epoch=1, epochs_per_cycle=10):
    if selection_methods is None:
        selection_methods = ['full', 'random', 'el2n', 'forget', 'craig']

    results = {}

    # 1. Prepare Backdoor Dataset
    # We no longer check for poison_type; we assume backdoor.
    ds_poisoned = new_backdoor_dataset(data_root, poison_frac=poison_frac, seed=seed)
    poisoned_set = ds_poisoned.poisoned_idx

    n = len(ds_poisoned)
    k = int(n * subset_frac)
    print(f'Running Backdoor, poison_frac={poison_frac}, subset size {k} (of {n})')

    for method in selection_methods:
        # Reset all seeds before each method for maximum reproducibility
        set_all_random_seeds(seed)

        print(f'\n---- Selection method: {method} ----')

        # 2. Select Indices
        if method == 'full':
            indices = list(range(n))
        elif method == 'random':
            indices = sample_random_indices(ds_poisoned, k, seed=seed)
        elif method == 'el2n':
            scores = compute_el2n_scores(get_model, ds_poisoned,
                                       epochs=el2n_epochs, device=device, num_workers=num_workers, seed=seed)
            indices = np.argsort(scores, kind='stable')[:k].tolist()
        elif method == 'forget':
            scores = compute_forgetting_scores(get_model, ds_poisoned,
                                             epochs=forget_epochs, device=device, num_workers=num_workers, seed=seed)
            indices = np.argsort(scores, kind='stable')[:k].tolist()
        elif method == 'craig':
            indices = select_subset_craig(get_model, ds_poisoned,
                                        subset_size=k, device=device, num_workers=num_workers,
                                        pretrain_epochs=craig_pretrain_epochs, seed=seed)

        kept, total_poison, retention = compute_poison_retention(poisoned_set, indices)
        print(f'Poison retention: {kept}/{total_poison} ({retention:.4f})')

        subset = subset_from_indices(ds_poisoned, indices)

        # 3. Train Dense Model with History Tracking
        model = get_model(device=device, seed=seed)
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_epochs, eta_min=1e-5)
        crit = nn.CrossEntropyLoss()

        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Use train_with_history wrapper to track losses
        model, history = train_with_history(
            model, train_loader, test_loader, opt, scheduler, crit,
            epochs=train_epochs, device=device
        )

        # 4. Evaluate Dense Model
        loss, acc = evaluate(model, test_loader, device=device)

        # Removed the 'if poison_type == ...' check
        asr = compute_backdoor_asr(model, test_loader, trigger_patch_size=6, target_label=0, device=device)

        print(f'Dense model - Clean accuracy: {acc:.4f}, ASR: {asr:.4f}')

        # 5. Pruning & Retraining
        print("Applying iterative magnitude pruning...")
        pruned_model, mask = iterative_magnitude_prune_and_retrain(
            get_model,
            subset,
            test_loader,
            fraction_to_prune=fraction_to_prune,
            iterations=iterations,
            rewind_epoch=rewind_epoch,
            epochs_per_cycle=epochs_per_cycle,
            num_workers=num_workers,
            device=device,
            seed=seed
        )

        ploss, pacc = evaluate(pruned_model, test_loader, device=device)
        pasr = compute_backdoor_asr(pruned_model, test_loader, trigger_patch_size=6, target_label=0, device=device)

        print(f'Pruned model - Clean accuracy: {pacc:.4f}, ASR: {pasr:.4f}')

        results[method] = {
            'subset_size': len(indices),
            'clean_acc': acc,
            'asr': asr,
            'poison_kept': kept,
            'poison_total': total_poison,
            'poison_retention': retention,
            'pruned_acc': pacc,
            'pruned_asr': pasr,
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs'],
            'pruned_test_loss': ploss
        }

    return results


def run_all_experiments(num_replicates=3, base_seed=0, poison_frac=0.02, subset_frac=0.25,
                       selection_methods=None, device='cuda', num_workers=4,
                       data_root='./data', train_epochs=50, batch_size=128, test_loader=None,
                       el2n_epochs=20, forget_epochs=20, craig_pretrain_epochs=10,
                       fraction_to_prune=0.3, iterations=2, rewind_epoch=1, epochs_per_cycle=10):
    """
    Run multiple replicates of run_all_methods and aggregate results.

    Args:
        num_replicates: Number of times to run the experiment
        base_seed: Starting seed value for deterministic seed generation
        ... (all other parameters from run_all_methods)

    Returns:
        aggregated_results: Dictionary with keys for each selection method, containing:
            - All individual replicate results
            - Aggregated statistics (mean, std) across replicates
    """
    if selection_methods is None:
        selection_methods = ['full', 'random', 'el2n', 'forget', 'craig']

    # Generate n_replicates random seeds deterministically from base_seed
    rng = np.random.RandomState(base_seed)
    replicate_seeds = rng.randint(0, 2**31 - 1, size=num_replicates).tolist()

    all_replicates = []

    print(f"\n{'='*80}")
    print(f"Running {num_replicates} replicates of the experiment")
    print(f"Base seed: {base_seed}")
    print(f"Replicate seeds: {replicate_seeds}")
    print(f"{'='*80}\n")

    for replicate_idx in range(num_replicates):
        seed = replicate_seeds[replicate_idx]
        print(f"\n{'='*80}")
        print(f"REPLICATE {replicate_idx + 1}/{num_replicates} (seed={seed})")
        print(f"{'='*80}\n")

        results = run_all_methods(
            poison_frac=poison_frac,
            subset_frac=subset_frac,
            selection_methods=selection_methods,
            device=device,
            num_workers=num_workers,
            seed=seed,
            data_root=data_root,
            train_epochs=train_epochs,
            batch_size=batch_size,
            test_loader=test_loader,
            el2n_epochs=el2n_epochs,
            forget_epochs=forget_epochs,
            craig_pretrain_epochs=craig_pretrain_epochs,
            fraction_to_prune=fraction_to_prune,
            iterations=iterations,
            rewind_epoch=rewind_epoch,
            epochs_per_cycle=epochs_per_cycle
        )

        all_replicates.append(results)

    # Aggregate results across replicates
    aggregated_results = {}

    for method in selection_methods:
        aggregated_results[method] = {
            'replicates': [],
            'mean': {},
            'std': {}
        }

        # Collect all replicate data for this method
        for rep_idx, rep_results in enumerate(all_replicates):
            aggregated_results[method]['replicates'].append({
                'replicate_idx': rep_idx,
                'seed': replicate_seeds[rep_idx],
                **rep_results[method]
            })

        # Compute statistics for scalar metrics
        scalar_metrics = ['clean_acc', 'asr', 'poison_retention', 'pruned_acc', 'pruned_asr', 'pruned_test_loss']

        for metric in scalar_metrics:
            values = [rep_results[method][metric] for rep_results in all_replicates]
            aggregated_results[method]['mean'][metric] = np.mean(values)
            aggregated_results[method]['std'][metric] = np.std(values)

        # Aggregate training histories (compute mean and std across replicates for each epoch)
        history_keys = ['train_losses', 'train_accs', 'test_losses', 'test_accs']

        for history_key in history_keys:
            # Stack all histories for this metric across replicates
            all_histories = [rep_results[method][history_key] for rep_results in all_replicates]

            # Convert to numpy array (replicates x epochs)
            histories_array = np.array(all_histories)

            # Compute mean and std across replicates for each epoch
            aggregated_results[method]['mean'][history_key] = np.mean(histories_array, axis=0).tolist()
            aggregated_results[method]['std'][history_key] = np.std(histories_array, axis=0).tolist()

    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS (across {num_replicates} replicates)")
    print(f"{'='*80}\n")

    for method in selection_methods:
        print(f"\n{method.upper()}:")
        print(f"  Clean Accuracy:      {aggregated_results[method]['mean']['clean_acc']:.4f} ± {aggregated_results[method]['std']['clean_acc']:.4f}")
        print(f"  ASR:                 {aggregated_results[method]['mean']['asr']:.4f} ± {aggregated_results[method]['std']['asr']:.4f}")
        print(f"  Poison Retention:    {aggregated_results[method]['mean']['poison_retention']:.4f} ± {aggregated_results[method]['std']['poison_retention']:.4f}")
        print(f"  Pruned Accuracy:     {aggregated_results[method]['mean']['pruned_acc']:.4f} ± {aggregated_results[method]['std']['pruned_acc']:.4f}")
        print(f"  Pruned ASR:          {aggregated_results[method]['mean']['pruned_asr']:.4f} ± {aggregated_results[method]['std']['pruned_asr']:.4f}")

    return aggregated_results