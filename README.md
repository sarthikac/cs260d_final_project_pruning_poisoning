# Mitigating Backdoor Poisoning Using Data Summarization and Model Pruning
Sarthika Chimmula, Demetri Nicolaou, Adi Pillai

This repository contains the code and experiments for our CS260D final project:  [**“Mitigating Backdoor Poisoning using Data Summarization and Model Pruning.”**](./COM%20SCI%20260D%20Final%20Report.pdf)  

We study whether lightweight, single-shot data summarization (coreset) methods combined with model pruning can mitigate backdoor poisoning attacks without requiring complex multi-stage poison detection.


## Overview

Backdoor poisoning attacks embed hidden triggers into training data, causing models to misclassify triggered inputs while retaining high clean accuracy. Traditional defenses often require repeatedly re-selecting data throughout training or performing explicit poison detection.

This project investigates whether static, pre-training data summarization methods—Random, EL2N, Forgetting Scores, and CRAIG—can naturally filter out poisoned samples, and whether iterative magnitude pruning (IMP) further improves robustness.

Our key finding:
CRAIG is the only method that significantly filters poisoned samples and meaningfully reduces the attack success rate (ASR), especially when combined with pruning.

## Repository Structure
```
cd260d_final_project_pruning_poisoning/
│
├── backdoor_aggregated_loss_history.csv      # Aggregated loss curves across replicates for backdoor experiments
├── backdoor_aggregated_results.csv           # Aggregated accuracy & ASR results across replicates
├── backdoor_all_replicates.csv               # Raw results for all poisoning replicates
├── backdoor_loss_history_prototype*.csv      # Prototype runs: loss curves for early/backbone experiments
├── backdoor_results_prototype*.csv           # Prototype runs: accuracy, ASR, poison retention
│
├── COM SCI 260D Final Report.pdf             # Final written report summarizing methods, results, analysis
├── README.md                                 # Project description, instructions, and findings
├── .gitignore                                # Files/folders excluded from version control
│
├── craig.py                                   # CRAIG subset-selection implementation
├── el2n.py                                    # EL2N scoring to measure example difficulty
├── forgetting.py                              # Forgetting event implementation and scoring
├── prune.py                                   # Pruning logic (IMP-style pruning, mask updates)
├── poison.py                                  # Backdoor poisoning (trigger insertion, poisoned data generation)
├── dataset.py                                 # CIFAR-10 dataset loading, preprocessing, and poison integration
├── model.py                                   # Model definitions (ResNet architecture, deterministic settings)
├── train_utils.py                             # Training utilities (loops, LR scheduling, checkpoints)
├── utils.py                                   # General utilities: seed control, logging, helper functions
├── GradualWarmupScheduler.py                  # Learning rate warmup scheduler implementation
│
├── results.py                                 # Aggregates outputs and writes CSV results tables
├── run.py                                     # Main experiment driver (poison → prune → train → evaluate)
│
├── final_poison_prune_selection_notebook.ipynb   # Jupyter notebook for experiment demos and analysis
│
```

## Project Goals
We evaluate whether:
- Data summarization alone can mitigate backdoor poisoning.
- Combining summarization with pruning reduces attack success rate (ASR).

## Methods Summary
### Backdoor Attack
- CIFAR-10 with 2% poisoned samples
- Red square trigger
- All poisoned samples relabeled to target class 0

### Data Summarization Methods
We evaluate five dataset variants at 50% downsampling for each:

| Method           | Description                               |
| ---------------- | ----------------------------------------- |
| Full             | Entire poisoned dataset                   |
| Random           | Random 50% subset                         |
| EL2N             | Select samples with low early-epoch loss  |
| Forgetting Score | Select stable (never-forgotten) samples   |
| CRAIG            | Gradient-based representativeness coreset |

### Model & Training
- ResNet-18 adapted for CIFAR-10
- SGD, lr=0.1, momentum=0.9, weight decay=5e-4
- Cosine annealing LR schedule
- CRAIG uses LR warmup (10 epochs)

### Pruning
- We apply Iterative Magnitude Pruning (IMP):
  - Rewind to epoch 1
  - Retrain to epoch 10
  - Prune 30% weights
  - Repeat 3 rounds
 
## Key Results
### Poison Retention Rate
- Full, Random, EL2N: ≈1.0 (retain almost all poisons)
- Forgetting: ~0.89
- CRAIG: 0.1414 ± 0.0609 → removes ≈86% of poisoned samples

### Attack Success Rate (ASR)
| Method                            | Dense ASR           | Pruned ASR          |
| --------------------------------- | ------------------- | ------------------- |
| Full / Random / EL2N / Forgetting | ≈1.0                | ≈1.0                |
| **CRAIG**                         | **0.8377 ± 0.1288** | **0.6045 ± 0.2815** |

### Clean Accuracy
- CRAIG maintains ~0.80 clean accuracy even after pruning.

## Results Summary
<img width="554" height="264" alt="Screenshot 2025-12-11 at 1 43 29 PM" src="https://github.com/user-attachments/assets/c580cd56-723a-4c6f-b95d-da57bd2dfcb1" />

<img width="596" height="257" alt="Screenshot 2025-12-11 at 1 44 02 PM" src="https://github.com/user-attachments/assets/f1665d47-da36-4fc5-b7ff-cd8420f6be8a" />


| Method     | Poison Ret. Rate       | Dense Acc.            | Pruned Acc.           | Dense ASR            | Pruned ASR           |
|------------|-------------------------|------------------------|------------------------|-----------------------|-----------------------|
| **Full**       | 1.0000±0.0000            | 0.9421±0.0021          | 0.8981±0.0084          | 0.9997±0.0002         | 0.9950±0.0027         |
| **Random**     | 0.9952±0.0016            | 0.9118±0.0011          | 0.8393±0.0059          | 0.9997±0.0002         | 0.9957±0.0026         |
| **EL2N**       | 0.9898±0.0055            | 0.8196±0.0071          | 0.7500±0.0100          | 0.9998±0.0002         | 0.9978±0.0022         |
| **Forgetting** | 0.8952±0.0408            | 0.8779±0.0026          | 0.7999±0.0219          | 0.9995±0.0004         | 0.9939±0.0027         |
| **CRAIG**      | 0.1414±0.0609            | 0.8664±0.0013          | 0.7983±0.0235          | 0.8377±0.1288         | 0.6045±0.2815         |

#### CRAIG is the only method providing meaningful backdoor mitigation.
