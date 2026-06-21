import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Base repository directory (two levels up from backpas/src/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKPAS_DIR = REPO_ROOT / "backpas"

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

log_dir = BACKPAS_DIR / "wkdir" / "14_bounded_strongly_correlated" / "ml_training" / "graph_with_literals_3_GTR"
files = [
    "training_log.csv",
]

run_names = [
    "Run 1 (Layers: 3, Train: 3986)",
]

dfs = []
for i, file_name in enumerate(files):
    path = log_dir / file_name
    df = pd.read_csv(path)
    df['Run_ID'] = i + 1
    df['Run_Name'] = run_names[i]
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

output_dir = BACKPAS_DIR / "analysis_results"
os.makedirs(output_dir, exist_ok=True)

# 1. Learning Curves
def plot_learning_curves(data, metric, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    sns.lineplot(data=data[data['partition'] == 'train'], x='epoch', y=metric, hue='Run_Name', ax=axes[0])
    axes[0].set_title(f'Train {title}')
    sns.lineplot(data=data[data['partition'] == 'valid'], x='epoch', y=metric, hue='Run_Name', ax=axes[1])
    axes[1].set_title(f'Valid {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_learning_curves(all_data, 'loss', 'Loss', '1_learning_curves_loss.png')
plot_learning_curves(all_data, 'f1_score_macro', 'Macro F1-Score', '1_learning_curves_f1.png')
plot_learning_curves(all_data, 'accuracy_macro', 'Macro Accuracy', '1_learning_curves_accuracy.png')

# 2. Ablation Study (Layers)
ablation_data = all_data[all_data['Run_ID'].isin([1, 2, 3])]
plot_learning_curves(ablation_data, 'loss', 'Loss (Ablation - Layers)', '2_ablation_layers_loss.png')
plot_learning_curves(ablation_data, 'f1_score_macro', 'Macro F1-Score (Ablation - Layers)', '2_ablation_layers_f1.png')

best_valid_ablation = ablation_data[ablation_data['partition'] == 'valid'].groupby('Run_Name')[['loss', 'f1_score_macro']].agg({'loss':'min', 'f1_score_macro':'max'}).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(data=best_valid_ablation, x='Run_Name', y='loss', ax=axes[0])
axes[0].set_title('Best Valid Loss (Ablation - Layers)')
sns.barplot(data=best_valid_ablation, x='Run_Name', y='f1_score_macro', ax=axes[1])
axes[1].set_title('Best Valid Macro F1 (Ablation - Layers)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_ablation_layers_bars.png'))
plt.close()

# 3. Scalability Study (Data Size)
scalability_data = all_data[all_data['Run_ID'].isin([3, 4, 5, 6])]
scalability_summary = scalability_data[scalability_data['partition'] == 'valid'].groupby('Run_Name').agg({
    'loss': 'min',
    'f1_score_macro': 'max',
    'Run_ID': 'first'
}).reset_index()

train_sizes = {3: 364, 4: 590, 5: 799, 6: 1699}
scalability_summary['Train_Size'] = scalability_summary['Run_ID'].map(train_sizes)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.lineplot(data=scalability_summary, x='Train_Size', y='loss', marker='o', ax=axes[0])
axes[0].set_title('Best Valid Loss vs Training Set Size')
sns.lineplot(data=scalability_summary, x='Train_Size', y='f1_score_macro', marker='o', ax=axes[1])
axes[1].set_title('Best Valid Macro F1 vs Training Set Size')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_scalability_data_size.png'))
plt.close()

# 4. Final Comparison
final_comparison = all_data[all_data['partition'] == 'valid'].groupby('Run_Name').agg({
    'f1_score_macro': 'max',
    'retrieval_precision_0.50': 'max'
}).reset_index()

final_melted = pd.melt(final_comparison, id_vars='Run_Name', var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(data=final_melted, x='Run_Name', y='Score', hue='Metric')
plt.title('Best Validation Metrics Across All Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_final_comparison.png'))
plt.close()

# 5. Additional Visualizations
time_summary = all_data[all_data['partition'] == 'train'].groupby('Run_Name')['time'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=time_summary, x='Run_Name', y='time')
plt.title('Average Epoch Training Time')
plt.ylabel('Seconds')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_epoch_time_comparison.png'))
plt.close()

print("Plots successfully generated in output directory:", output_dir)
