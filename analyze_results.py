import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# Read CSV file
df = pd.read_csv('/Users/ooongs/Github/IPCIR/data.csv')

# Function to parse the Results column
def parse_results(results_text):
    """Parse the multi-line results text into a dictionary"""
    data = {}

    # Extract Recall metrics
    recall_pattern = r'Recall@(\d+): ([\d.]+)'
    for match in re.finditer(recall_pattern, results_text):
        k = match.group(1)
        value = float(match.group(2))
        data[f'Recall@{k}'] = value

    # Extract mAP metrics
    map_pattern = r'mAP@(\d+): ([\d.]+)'
    for match in re.finditer(map_pattern, results_text):
        k = match.group(1)
        value = float(match.group(2))
        data[f'mAP@{k}'] = value

    # Extract semantic aspect metrics
    aspects = [
        'Cardinality', 'Addition', 'Negation', 'Direct Addressing',
        'Compare & Change', 'Comparative Statement',
        'Statement with Conjunction', 'Spatial Relations & Background', 'Viewpoint'
    ]

    for aspect in aspects:
        pattern = rf'{aspect}: ([\d.]+)'
        match = re.search(pattern, results_text)
        if match:
            data[f'Aspect_{aspect}'] = float(match.group(1))

    return data

# Parse all results
parsed_data = []
for idx, row in df.iterrows():
    parsed = parse_results(row['Results'])
    parsed['Model'] = row['Model']
    parsed['Inference_Step'] = int(row['Inference Step'])
    parsed['Scale'] = float(row['Scale'])
    parsed_data.append(parsed)

# Create a clean DataFrame
df_clean = pd.DataFrame(parsed_data)

# Save the cleaned data
df_clean.to_csv('/Users/ooongs/Github/IPCIR/data_cleaned.csv', index=False)
print("✓ Cleaned data saved to data_cleaned.csv")
print(f"\nDataset shape: {df_clean.shape}")
print(f"Inference Steps: {sorted(df_clean['Inference_Step'].unique())}")
print(f"Scale range: {df_clean['Scale'].min()} - {df_clean['Scale'].max()}")
print(f"\nColumns: {list(df_clean.columns)}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('mAP Performance Analysis: Inference Steps vs Scale', fontsize=16, fontweight='bold')

inference_steps = sorted(df_clean['Inference_Step'].unique())
colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c'}
markers = {1: 'o', 2: 's', 4: '^'}

# Plot mAP@5, mAP@10, mAP@25, mAP@50
map_metrics = ['mAP@5', 'mAP@10', 'mAP@25', 'mAP@50']

for idx, metric in enumerate(map_metrics):
    ax = axes[idx // 2, idx % 2]

    for step in inference_steps:
        step_data = df_clean[df_clean['Inference_Step'] == step].sort_values('Scale')
        ax.plot(step_data['Scale'], step_data[metric],
                label=f'Step {step}',
                color=colors[step],
                marker=markers[step],
                linewidth=2,
                markersize=6,
                alpha=0.8)

    ax.set_xlabel('Scale', fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(f'{metric} vs Scale', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('/Users/ooongs/Github/IPCIR/map_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: map_analysis.png")

# Additional plot: All mAP metrics together for each inference step
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('All mAP Metrics by Inference Step', fontsize=16, fontweight='bold')

for idx, step in enumerate(inference_steps):
    ax = axes2[idx]
    step_data = df_clean[df_clean['Inference_Step'] == step].sort_values('Scale')

    for metric in map_metrics:
        ax.plot(step_data['Scale'], step_data[metric],
                label=metric,
                linewidth=2,
                marker='o',
                markersize=5)

    ax.set_xlabel('Scale', fontsize=11, fontweight='bold')
    ax.set_ylabel('mAP Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Inference Step {step}', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('/Users/ooongs/Github/IPCIR/map_by_step.png', dpi=300, bbox_inches='tight')
print("✓ Saved: map_by_step.png")

# Heatmap for mAP@10 (most common metric)
fig3, ax3 = plt.subplots(figsize=(12, 6))

pivot_data = df_clean.pivot(index='Inference_Step', columns='Scale', values='mAP@10')
im = ax3.imshow(pivot_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')

ax3.set_xticks(range(len(pivot_data.columns)))
ax3.set_xticklabels([f'{x:.1f}' for x in pivot_data.columns], rotation=45)
ax3.set_yticks(range(len(pivot_data.index)))
ax3.set_yticklabels(pivot_data.index)

ax3.set_xlabel('Scale', fontsize=12, fontweight='bold')
ax3.set_ylabel('Inference Step', fontsize=12, fontweight='bold')
ax3.set_title('mAP@10 Heatmap: Inference Steps vs Scale', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('mAP@10', fontsize=11, fontweight='bold')

# Add text annotations
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        text = ax3.text(j, i, f'{pivot_data.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('/Users/ooongs/Github/IPCIR/map10_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: map10_heatmap.png")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

for step in inference_steps:
    step_data = df_clean[df_clean['Inference_Step'] == step]
    print(f"\n📊 Inference Step {step}:")
    print(f"   Best mAP@10: {step_data['mAP@10'].max():.2f} (Scale={step_data.loc[step_data['mAP@10'].idxmax(), 'Scale']:.1f})")
    print(f"   Worst mAP@10: {step_data['mAP@10'].min():.2f} (Scale={step_data.loc[step_data['mAP@10'].idxmin(), 'Scale']:.1f})")
    print(f"   Mean mAP@10: {step_data['mAP@10'].mean():.2f}")
    print(f"   Std mAP@10: {step_data['mAP@10'].std():.2f}")

# Find optimal scale for each inference step
print("\n" + "="*60)
print("OPTIMAL SCALE FOR EACH INFERENCE STEP")
print("="*60)
for step in inference_steps:
    step_data = df_clean[df_clean['Inference_Step'] == step]
    best_idx = step_data['mAP@10'].idxmax()
    best_scale = step_data.loc[best_idx, 'Scale']
    best_map10 = step_data.loc[best_idx, 'mAP@10']
    print(f"\n🎯 Step {step}: Scale = {best_scale:.1f}, mAP@10 = {best_map10:.2f}")

print("\n✅ Analysis complete! Generated 3 visualization files.")
