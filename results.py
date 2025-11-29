import matplotlib.pyplot as plt
import pandas as pd

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
    rows = []
    for m,v in results.items():
        r = {'method': m}
        r.update(v)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(fname, index=False)
    print('Saved results to', fname)
