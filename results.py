import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_loss_curves(results, title='Loss Curves'):
    """
    Plot training and test loss curves for all methods.

    Args:
        results: dict with method names as keys, each containing:
                 'train_losses': list of per-epoch training losses
                 'test_losses': list of per-epoch test losses
        title: plot title
    """
    methods = list(results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss curves
    for method in methods:
        if 'train_losses' in results[method] and len(results[method]['train_losses']) > 0:
            epochs = range(1, len(results[method]['train_losses']) + 1)
            ax1.plot(epochs, results[method]['train_losses'], marker='o', label=method)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test loss curves
    for method in methods:
        if 'test_losses' in results[method] and len(results[method]['test_losses']) > 0:
            epochs = range(1, len(results[method]['test_losses']) + 1)
            ax2.plot(epochs, results[method]['test_losses'], marker='o', label=method)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss per Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_results(results, title='Results'):
    methods = list(results.keys())
    clean = [results[m]['clean_acc'] for m in methods]
    asr = [results[m]['asr'] for m in methods]
    retention = [results[m]['poison_retention'] for m in methods]
    pruned_acc = [results[m]['pruned_acc'] for m in methods]
    pruned_asr = [results[m]['pruned_asr'] for m in methods]

    x = np.arange(len(methods))
    width = 0.28
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.bar(x - width, clean, width, label='Clean Acc')
    plt.bar(x, pruned_acc, width, label='Pruned Acc')
    plt.xticks(x, methods, rotation=30)
    plt.ylabel('Accuracy'); plt.ylim(0,1); plt.legend()
    plt.title('Clean Acc (Dense vs Pruned)')

    plt.subplot(1,2,2)
    plt.bar(x - width, asr, width, label='ASR (dense)')
    plt.bar(x, pruned_asr, width, label='ASR (pruned)')
    plt.xticks(x, methods, rotation=30)
    plt.ylabel('Attack Success Rate'); plt.ylim(0,1); plt.legend()
    plt.title('Attack Success Rate (Dense vs Pruned)')
    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    plt.figure(figsize=(6,3))
    plt.bar(methods, retention)
    plt.title('Poison Retention Fraction in Selected Subsets')
    plt.ylim(0,1)
    plt.show()

def save_results_csv(results, fname='results_summary.csv'):
    """
    Save results summary to CSV. Handles optional loss columns.

    Expected columns:
    - method, subset_size, clean_acc, pruned_acc
    - asr, pruned_asr, poison_kept, poison_total, poison_retention (if poisoned)
    - final_train_loss, final_test_loss, pruned_test_loss (if tracked)
    """
    rows = []
    for m, v in results.items():
        r = {'method': m}
        # Core metrics
        r['subset_size'] = v.get('subset_size', 0)
        r['clean_acc'] = v.get('clean_acc', 0.0)
        r['pruned_acc'] = v.get('pruned_acc', 0.0)

        # Poison metrics (optional)
        r['asr'] = v.get('asr', 0.0)
        r['pruned_asr'] = v.get('pruned_asr', 0.0)
        r['poison_kept'] = v.get('poison_kept', 0)
        r['poison_total'] = v.get('poison_total', 0)
        r['poison_retention'] = v.get('poison_retention', 0.0)

        # Loss metrics (optional)
        if 'train_losses' in v and len(v['train_losses']) > 0:
            r['final_train_loss'] = v['train_losses'][-1]
        if 'test_losses' in v and len(v['test_losses']) > 0:
            r['final_test_loss'] = v['test_losses'][-1]
        if 'pruned_test_loss' in v:
            r['pruned_test_loss'] = v['pruned_test_loss']

        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)
    print('Saved results to', fname)

def save_loss_history_csv(results, fname='loss_history.csv'):
    """
    Save per-epoch loss and accuracy history to CSV.

    Format: method, epoch, train_loss, train_acc, test_loss, test_acc
    """
    rows = []
    for method, v in results.items():
        if 'train_losses' not in v or 'test_losses' not in v:
            continue

        train_losses = v.get('train_losses', [])
        train_accs = v.get('train_accs', [])
        test_losses = v.get('test_losses', [])
        test_accs = v.get('test_accs', [])

        num_epochs = len(train_losses)
        for epoch in range(num_epochs):
            row = {
                'method': method,
                'epoch': epoch + 1,
                'train_loss': train_losses[epoch] if epoch < len(train_losses) else None,
                'train_acc': train_accs[epoch] if epoch < len(train_accs) else None,
                'test_loss': test_losses[epoch] if epoch < len(test_losses) else None,
                'test_acc': test_accs[epoch] if epoch < len(test_accs) else None
            }
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(fname, index=False)
        print('Saved loss history to', fname)
    else:
        print('No loss history to save')


# ==================================================================================
# NEW FUNCTIONS FOR AGGREGATED RESULTS (with mean and std from multiple replicates)
# ==================================================================================

def plot_aggregated_loss_curves(aggregated_results, title='Loss Curves (Mean ± Std)'):
    """
    Plot training and test loss curves with mean and standard deviation shaded regions.

    Args:
        aggregated_results: dict with method names as keys, each containing:
                            'mean': {'train_losses': [...], 'test_losses': [...]}
                            'std': {'train_losses': [...], 'test_losses': [...]}
        title: plot title
    """
    methods = list(aggregated_results.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss curves
    for method in methods:
        if 'mean' in aggregated_results[method] and 'train_losses' in aggregated_results[method]['mean']:
            mean_losses = aggregated_results[method]['mean']['train_losses']
            std_losses = aggregated_results[method]['std']['train_losses']
            epochs = range(1, len(mean_losses) + 1)

            line, = ax1.plot(epochs, mean_losses, marker='o', label=method)
            ax1.fill_between(epochs,
                            np.array(mean_losses) - np.array(std_losses),
                            np.array(mean_losses) + np.array(std_losses),
                            alpha=0.2, color=line.get_color())

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test loss curves
    for method in methods:
        if 'mean' in aggregated_results[method] and 'test_losses' in aggregated_results[method]['mean']:
            mean_losses = aggregated_results[method]['mean']['test_losses']
            std_losses = aggregated_results[method]['std']['test_losses']
            epochs = range(1, len(mean_losses) + 1)

            line, = ax2.plot(epochs, mean_losses, marker='o', label=method)
            ax2.fill_between(epochs,
                            np.array(mean_losses) - np.array(std_losses),
                            np.array(mean_losses) + np.array(std_losses),
                            alpha=0.2, color=line.get_color())

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss per Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_aggregated_results(aggregated_results, title='Results (Mean ± Std)'):
    """
    Plot aggregated results with error bars showing standard deviation.

    Args:
        aggregated_results: dict with method names as keys, each containing:
                            'mean': {'clean_acc': x, 'asr': y, ...}
                            'std': {'clean_acc': x, 'asr': y, ...}
        title: plot title
    """
    methods = list(aggregated_results.keys())

    clean_mean = [aggregated_results[m]['mean']['clean_acc'] for m in methods]
    clean_std = [aggregated_results[m]['std']['clean_acc'] for m in methods]

    asr_mean = [aggregated_results[m]['mean']['asr'] for m in methods]
    asr_std = [aggregated_results[m]['std']['asr'] for m in methods]

    retention_mean = [aggregated_results[m]['mean']['poison_retention'] for m in methods]
    retention_std = [aggregated_results[m]['std']['poison_retention'] for m in methods]

    pruned_acc_mean = [aggregated_results[m]['mean']['pruned_acc'] for m in methods]
    pruned_acc_std = [aggregated_results[m]['std']['pruned_acc'] for m in methods]

    pruned_asr_mean = [aggregated_results[m]['mean']['pruned_asr'] for m in methods]
    pruned_asr_std = [aggregated_results[m]['std']['pruned_asr'] for m in methods]

    x = np.arange(len(methods))
    width = 0.28

    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy (Dense vs Pruned)
    plt.subplot(1, 2, 1)
    plt.bar(x - width, clean_mean, width, yerr=clean_std, capsize=5, label='Clean Acc (Dense)')
    plt.bar(x, pruned_acc_mean, width, yerr=pruned_acc_std, capsize=5, label='Clean Acc (Pruned)')
    plt.xticks(x, methods, rotation=30, ha='right')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Clean Accuracy (Dense vs Pruned)')
    plt.grid(True, alpha=0.3, axis='y')

    # Subplot 2: ASR (Dense vs Pruned)
    plt.subplot(1, 2, 2)
    plt.bar(x - width, asr_mean, width, yerr=asr_std, capsize=5, label='ASR (Dense)')
    plt.bar(x, pruned_asr_mean, width, yerr=pruned_asr_std, capsize=5, label='ASR (Pruned)')
    plt.xticks(x, methods, rotation=30, ha='right')
    plt.ylabel('Attack Success Rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Attack Success Rate (Dense vs Pruned)')
    plt.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Separate plot for poison retention
    plt.figure(figsize=(8, 4))
    plt.bar(methods, retention_mean, yerr=retention_std, capsize=5, color='steelblue', alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Poison Retention Fraction')
    plt.title('Poison Retention in Selected Subsets')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def save_aggregated_results_csv(aggregated_results, fname='aggregated_results_summary.csv'):
    """
    Save aggregated results (mean and std) to CSV.

    Args:
        aggregated_results: dict with method names as keys, each containing:
                            'mean': {...metrics...}
                            'std': {...metrics...}
        fname: output CSV filename
    """
    rows = []
    for method, data in aggregated_results.items():
        mean_data = data['mean']
        std_data = data['std']

        row = {
            'method': method,
            'clean_acc_mean': mean_data.get('clean_acc', 0.0),
            'clean_acc_std': std_data.get('clean_acc', 0.0),
            'asr_mean': mean_data.get('asr', 0.0),
            'asr_std': std_data.get('asr', 0.0),
            'poison_retention_mean': mean_data.get('poison_retention', 0.0),
            'poison_retention_std': std_data.get('poison_retention', 0.0),
            'pruned_acc_mean': mean_data.get('pruned_acc', 0.0),
            'pruned_acc_std': std_data.get('pruned_acc', 0.0),
            'pruned_asr_mean': mean_data.get('pruned_asr', 0.0),
            'pruned_asr_std': std_data.get('pruned_asr', 0.0),
            'pruned_test_loss_mean': mean_data.get('pruned_test_loss', 0.0),
            'pruned_test_loss_std': std_data.get('pruned_test_loss', 0.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)
    print(f'Saved aggregated results to {fname}')


def save_aggregated_loss_history_csv(aggregated_results, fname='aggregated_loss_history.csv'):
    """
    Save per-epoch loss and accuracy history (mean and std) to CSV.

    Format: method, epoch, train_loss_mean, train_loss_std, train_acc_mean, train_acc_std,
            test_loss_mean, test_loss_std, test_acc_mean, test_acc_std
    """
    rows = []
    for method, data in aggregated_results.items():
        if 'mean' not in data or 'train_losses' not in data['mean']:
            continue

        mean_data = data['mean']
        std_data = data['std']

        train_losses_mean = mean_data.get('train_losses', [])
        train_losses_std = std_data.get('train_losses', [])
        train_accs_mean = mean_data.get('train_accs', [])
        train_accs_std = std_data.get('train_accs', [])
        test_losses_mean = mean_data.get('test_losses', [])
        test_losses_std = std_data.get('test_losses', [])
        test_accs_mean = mean_data.get('test_accs', [])
        test_accs_std = std_data.get('test_accs', [])

        num_epochs = len(train_losses_mean)
        for epoch in range(num_epochs):
            row = {
                'method': method,
                'epoch': epoch + 1,
                'train_loss_mean': train_losses_mean[epoch] if epoch < len(train_losses_mean) else None,
                'train_loss_std': train_losses_std[epoch] if epoch < len(train_losses_std) else None,
                'train_acc_mean': train_accs_mean[epoch] if epoch < len(train_accs_mean) else None,
                'train_acc_std': train_accs_std[epoch] if epoch < len(train_accs_std) else None,
                'test_loss_mean': test_losses_mean[epoch] if epoch < len(test_losses_mean) else None,
                'test_loss_std': test_losses_std[epoch] if epoch < len(test_losses_std) else None,
                'test_acc_mean': test_accs_mean[epoch] if epoch < len(test_accs_mean) else None,
                'test_acc_std': test_accs_std[epoch] if epoch < len(test_accs_std) else None,
            }
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(fname, index=False)
        print(f'Saved aggregated loss history to {fname}')
    else:
        print('No aggregated loss history to save')


def save_all_replicates_csv(aggregated_results, fname='all_replicates_results.csv'):
    """
    Save all individual replicate results to a single CSV file.

    Format: method, replicate_idx, seed, clean_acc, asr, poison_retention, pruned_acc, pruned_asr, etc.
    """
    rows = []
    for method, data in aggregated_results.items():
        if 'replicates' not in data:
            continue

        for rep_data in data['replicates']:
            row = {
                'method': method,
                'replicate_idx': rep_data.get('replicate_idx', 0),
                'seed': rep_data.get('seed', 0),
                'subset_size': rep_data.get('subset_size', 0),
                'clean_acc': rep_data.get('clean_acc', 0.0),
                'asr': rep_data.get('asr', 0.0),
                'poison_kept': rep_data.get('poison_kept', 0),
                'poison_total': rep_data.get('poison_total', 0),
                'poison_retention': rep_data.get('poison_retention', 0.0),
                'pruned_acc': rep_data.get('pruned_acc', 0.0),
                'pruned_asr': rep_data.get('pruned_asr', 0.0),
                'pruned_test_loss': rep_data.get('pruned_test_loss', 0.0),
            }
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(fname, index=False)
        print(f'Saved all replicate results to {fname}')
    else:
        print('No replicate data to save')
