import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Base repository directory (two levels up from backpas/src/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKPAS_DIR = REPO_ROOT / "backpas"

# Route configuration
CSV_PATH = BACKPAS_DIR / "dataset" / "backbone_summary.csv"
OUTPUT_IMAGE = BACKPAS_DIR / "dataset" / "backbone_distribution.png"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV file not found at: {CSV_PATH}")
        return

    # Load data
    df = pd.read_csv(CSV_PATH)
    
    # Check that the required columns exist
    required_cols = ['backbone_percentage', 'positive_backbone_percentage', 'negative_backbone_percentage']
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Column '{col}' not found in the CSV.")
            return

    # Configure chart styles
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 3 subplots for better understanding
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Distribution of the total backbone percentage
    sns.histplot(data=df, x='backbone_percentage', bins=30, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution of Backbone Percentage\n(Relative to size N)')
    axes[0].set_xlabel('Backbone Percentage (%)')
    axes[0].set_ylabel('Frequency (Num. of Instances)')

    # 2. Composition of Polos (Positives vs Negatives) inside the backbone
    sns.histplot(data=df, x='positive_backbone_percentage', bins=30, kde=True, ax=axes[1], color='lightgreen', label='Positives', alpha=0.6)
    sns.histplot(data=df, x='negative_backbone_percentage', bins=30, kde=True, ax=axes[1], color='salmon', label='Negatives', alpha=0.6)
    axes[1].set_title('Internal Composition of the Backbone\n(Proportion of Positives vs Negatives)')
    axes[1].set_xlabel('Proportion within the Backbone (0 to 1)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # 3. Relationship between backbone size and positive percentage
    # To see if very large backbones tend to be mostly positive or negative
    sns.scatterplot(data=df, x='backbone_percentage', y='positive_backbone_percentage', ax=axes[2], alpha=0.5, color='purple')
    axes[2].set_title('Relationship: % Total vs % Positive')
    axes[2].set_xlabel('Total Backbone Percentage')
    axes[2].set_ylabel('Positive Proportion of the Backbone')

    plt.tight_layout()
    
    # Save chart
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"=== Visualization generated successfully ===")
    print(f"- Image saved to: {OUTPUT_IMAGE}")
    
    # plt.show() # Uncomment if you want to see the pop-up window when running the script

if __name__ == "__main__":
    main()
