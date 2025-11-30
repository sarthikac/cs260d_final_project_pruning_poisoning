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
