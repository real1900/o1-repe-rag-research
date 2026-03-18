import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['HotpotQA\n(Multi-Hop Distractor)', 'HotpotQA\n(True Context Upper Bound)', 'SQuAD v1.1\n(Single-Hop Extraction)']
accuracies = [15.00, 40.00, 80.00]
colors = ['#ff6b6b', '#ffd93d', '#4d96ff']

# Set style
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

# Create bars
bars = ax.bar(datasets, accuracies, color=colors, width=0.6)

# Add numeric labels on top
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold',
                fontsize=12)

# Styling and Labels
ax.set_ylim(0, 100)
ax.set_ylabel('Accuracy (Normalized Substring Match)', fontsize=12, fontweight='bold', labelpad=15)
ax.set_title('Impact of Dataset Complexity on Llama-3.2-1B-Instruct Baseline', fontsize=14, fontweight='bold', pad=20)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')

# Add subtle grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
ax.set_axisbelow(True)

# Tweak tick labels
plt.xticks(fontsize=11)
plt.yticks(np.arange(0, 101, 20), fontsize=10)

plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
print("Saved baseline_comparison.png")
