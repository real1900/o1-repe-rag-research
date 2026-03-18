import matplotlib.pyplot as plt
import seaborn as sns

# Set headless backend
plt.switch_backend('Agg')

# Styling - Professional Academic
plt.figure(figsize=(8, 6))
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Data for Relative Improvement
labels = ['Baseline Blind RAG', 'O(1) RepE Steering']
accuracy = [4.0, 6.0]
# Professional color palette (Deep navy and muted teal)
colors = ['#34495E', '#16A085']

# Create Bar Plot
bars = plt.bar(labels, accuracy, color=colors, width=0.55, edgecolor='none')

# Emphasize the +50% Relative Jump
plt.title('Relative Answer F1 Improvement\n(Exact Match on HotpotQA)', fontsize=16, fontweight='bold', pad=30, color='#2C3E50')
plt.ylabel('F1 Accuracy Score (%)', fontsize=13, fontweight='bold', color='#2C3E50')
plt.ylim(0, 8)

# Add Data Labels inside the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval - 0.5, f'{yval}%', 
             ha='center', va='top', fontsize=14, fontweight='bold', color='white')

# Professional Annotation bracket for the 50% relative increase
x1 = bars[0].get_x() + bars[0].get_width() / 2.0
x2 = bars[1].get_x() + bars[1].get_width() / 2.0
y = max(accuracy) + 0.4
h = 0.2

# Draw the bracket
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='#2C3E50')
plt.text((x1+x2)/2.0, y+h+0.1, '+50% Relative Improvement', ha='center', va='bottom', 
         color='#2C3E50', fontsize=14, fontweight='bold')

# Clean up axes
plt.xticks(fontsize=12, fontweight='bold', color='#2C3E50')
plt.yticks(fontsize=11, color='#2C3E50')
plt.grid(axis='y', linestyle='-', alpha=0.3, color='#BDC3C7')
sns.despine(left=True, bottom=False)

# Save the figure
plt.tight_layout()
plt.savefig('relative_improvement.png', dpi=300, transparent=False)
print("Generated relative_improvement.png")
plt.close()
