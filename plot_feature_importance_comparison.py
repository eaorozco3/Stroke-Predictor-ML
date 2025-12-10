import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def plot_feature_importance_comparison(models_dict, X_train, y_train, X_test, y_test, feature_names):
    importance_df = pd.DataFrame(index=feature_names)
    
    for model_name, model in models_dict.items():
        print(f"\nCalculating importance for {model_name}...")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df[model_name] = importances
        
        elif hasattr(model, 'coef_'):
            # Use absolute values of coefficients
            importances = np.abs(model.coef_[0])
            importance_df[model_name] = importances
        
        else:
            result = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10, 
                random_state=42, 
                n_jobs=-1
            )

            importances = result.importances_mean
            importance_df[model_name] = importances
    
    importance_df_normalized = importance_df.div(importance_df.max(axis=0), axis=1)
    
    # Plot 1: Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(14, 8))
    importance_df_normalized.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Normalized Importance', fontsize=12)
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(importance_df_normalized.T, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(models_dict)))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels(models_dict.keys())
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Importance', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(len(models_dict)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{importance_df_normalized.iloc[j, i]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Feature Importance Heatmap Across Models', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Top 10 Features Average Importance
    importance_df['Average'] = importance_df.mean(axis=1)
    top_features = importance_df.nlargest(10, 'Average')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features['Average'].plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Top 10 Most Important Features (Average Across Models)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('top_10_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return the dataframe for further analysis
    return importance_df
