import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set headless backend for saving files
plt.switch_backend('Agg')

# 1. Bar Charts for Phase 4 Evaluation
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy Plot
labels = ['Baseline Blind RAG', 'O(1) RepE Steering']
accuracy = [4.00, 6.00]
colors = ['#FF6B6B', '#4ECDC4']

bars1 = ax[0].bar(labels, accuracy, color=colors)
ax[0].set_ylabel('F1 Exact Match Accuracy (%)', fontsize=12)
ax[0].set_title('Retrieval-Augmented Accuracy\n(Higher is Better)', fontsize=14, fontweight='bold')
ax[0].set_ylim(0, 8)
ax[0].grid(axis='y', linestyle='--', alpha=0.7)

# Add data labels
for bar in bars1:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Latency Plot
latency = [0.1883, 0.1700]
bars2 = ax[1].bar(labels, latency, color=colors)
ax[1].set_ylabel('Average Query Latency (Seconds)', fontsize=12)
ax[1].set_title('Inference Latency Bloat\n(Lower is Better)', fontsize=14, fontweight='bold')
ax[1].set_ylim(0, 0.25)
ax[1].grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.4f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300)
print("Generated performance_metrics.png")
plt.close()

# 2. Scatter Plot of the O(1) Probe Dataset
if os.path.exists('synthetic_alpha_tuning.csv'):
    df = pd.read_csv('synthetic_alpha_tuning.csv')
    df_success = df[(df['Success'] == True) & (df['Optimal_Alpha'] > 0)]
    
    if not df_success.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_success, 
            x='Token_Confidence', 
            y='Collapse_Alpha', 
            hue='Optimal_Alpha', 
            palette='viridis', 
            size='Prompt_Length',
            sizes=(20, 200),
            alpha=0.8
        )
        plt.title('O(1) Alpha Geometric Training Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Baseline Token Confidence (Max Logit Prob)', fontsize=12)
        plt.ylabel('Calculated Geometric Ceiling (Collapse Alpha)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('probe_distribution.png', dpi=300)
        print("Generated probe_distribution.png")
        plt.close()
