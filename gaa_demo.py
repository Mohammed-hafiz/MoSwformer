"""
Example usage of Graph Attention Autoencoder for Multi-Omics Data Analysis

This script demonstrates how to:
1. Load and use the trained GAA model
2. Extract latent representations from multi-omics data
3. Visualize attention weights
4. Use latent features for downstream analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import pickle
import os

from dataloader import data_set
from graph_attention_autoencoder import GraphAttentionAutoencoder
from config.gaa_config import GAAConfig


def load_trained_model(model_path: str, config: GAAConfig, input_dims: list):
    """Load a trained GAA model"""
    model = GraphAttentionAutoencoder(
        input_dims=input_dims,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        fusion_type='attention',
        graph_config={
            'k_neighbors': config.k_neighbors,
            'graph_type': config.graph_type,
            'similarity_method': config.similarity_method
        }
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def extract_all_latent_features(model: GraphAttentionAutoencoder, dataset: data_set):
    """Extract latent features for all samples in dataset"""
    latent_features = []
    sample_ids = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            data_list = [
                sample['data1'].unsqueeze(0),
                sample['data2'].unsqueeze(0), 
                sample['data3'].unsqueeze(0)
            ]
            
            # Extract latent representation
            latent = model.get_latent_representation(data_list)
            latent_features.append(latent.numpy().flatten())
            sample_ids.append(f'sample_{i}')
            labels.append(sample['label'].item())
    
    return np.array(latent_features), sample_ids, labels


def visualize_latent_space(latent_features: np.ndarray, labels: list, save_path: str = None):
    """Visualize latent space using PCA"""
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_features)
    
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                   c=[colors[i]], label=f'Class {label}', alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('GAA Latent Space Visualization (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return pca


def analyze_clustering_performance(latent_features: np.ndarray, true_labels: list):
    """Analyze clustering performance in latent space"""
    results = {}
    
    # Try different numbers of clusters
    n_clusters_range = range(2, min(10, len(set(true_labels)) + 3))
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(latent_features)
        
        # Calculate metrics
        ari = adjusted_rand_score(true_labels, predicted_labels)
        silhouette = silhouette_score(latent_features, predicted_labels)
        
        results[n_clusters] = {
            'ari': ari,
            'silhouette': silhouette,
            'predicted_labels': predicted_labels
        }
    
    return results


def analyze_attention_weights(attention_weights_path: str, modality_names: list = None):
    """Analyze and summarize attention weights"""
    if not os.path.exists(attention_weights_path):
        print(f"Attention weights file not found: {attention_weights_path}")
        return None
    
    with open(attention_weights_path, 'rb') as f:
        all_attention_weights = pickle.load(f)
    
    if not modality_names:
        modality_names = [f'Modality_{i}' for i in range(len(all_attention_weights[0]))]
    
    print("Attention Analysis:")
    print("=" * 50)
    
    # Analyze attention statistics for each modality
    for mod_idx, mod_name in enumerate(modality_names):
        print(f"\n{mod_name}:")
        
        # Collect attention values across all samples
        all_attn_values = []
        for sample_attn in all_attention_weights:
            modality_attn = sample_attn[mod_idx]  # List of attention weights for this modality
            for layer_attn in modality_attn:
                if len(layer_attn) == 2:  # (edge_index, attention_values)
                    attn_values = layer_attn[1]
                    all_attn_values.extend(attn_values.flatten().tolist())
        
        if all_attn_values:
            attn_array = np.array(all_attn_values)
            print(f"  Mean attention: {attn_array.mean():.4f}")
            print(f"  Std attention: {attn_array.std():.4f}")
            print(f"  Min attention: {attn_array.min():.4f}")
            print(f"  Max attention: {attn_array.max():.4f}")
    
    return all_attention_weights


def demonstrate_gaa_usage():
    """Complete demonstration of GAA usage"""
    print("Graph Attention Autoencoder Demo")
    print("=" * 50)
    
    # Load test data
    test_dataset = data_set('./BRCA_split/BRCA', train_flag=False)
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Check if trained model exists
    model_path = './test_gaa_outputs/best_gaa_model.pth'
    if not os.path.exists(model_path):
        print("No trained model found. Please run training first.")
        return
    
    # Configuration used for training
    config = GAAConfig(
        latent_dim=4,
        hidden_dim=8,
        num_heads=1,
        num_layers=1,
        k_neighbors=3,
        graph_type='feature',
        similarity_method='cosine'
    )
    
    # Get input dimensions from a sample
    sample = test_dataset[0]
    input_dims = [sample['data1'].shape[0], sample['data2'].shape[0], sample['data3'].shape[0]]
    
    # Load trained model
    print("Loading trained model...")
    model = load_trained_model(model_path, config, input_dims)
    
    # Extract latent features
    print("Extracting latent features...")
    latent_features, sample_ids, labels = extract_all_latent_features(model, test_dataset)
    
    print(f"Extracted latent features shape: {latent_features.shape}")
    print(f"Number of unique classes: {len(set(labels))}")
    
    # Create output directory
    demo_output_dir = './gaa_demo_outputs'
    os.makedirs(demo_output_dir, exist_ok=True)
    
    # Visualize latent space
    print("Creating latent space visualization...")
    pca = visualize_latent_space(latent_features, labels, 
                                os.path.join(demo_output_dir, 'latent_space_pca.png'))
    
    # Analyze clustering performance
    print("Analyzing clustering performance...")
    clustering_results = analyze_clustering_performance(latent_features, labels)
    
    # Find best clustering result
    best_k = max(clustering_results.keys(), key=lambda k: clustering_results[k]['ari'])
    best_result = clustering_results[best_k]
    
    print(f"\nBest clustering result (k={best_k}):")
    print(f"  Adjusted Rand Index: {best_result['ari']:.4f}")
    print(f"  Silhouette Score: {best_result['silhouette']:.4f}")
    
    # Save clustering results
    clustering_df = pd.DataFrame({
        'k': list(clustering_results.keys()),
        'ari': [r['ari'] for r in clustering_results.values()],
        'silhouette': [r['silhouette'] for r in clustering_results.values()]
    })
    clustering_df.to_csv(os.path.join(demo_output_dir, 'clustering_results.csv'), index=False)
    
    # Save latent features
    latent_df = pd.DataFrame(latent_features, index=sample_ids)
    latent_df['true_label'] = labels
    latent_df['predicted_cluster'] = best_result['predicted_labels']
    latent_df.to_csv(os.path.join(demo_output_dir, 'latent_features_analysis.csv'))
    
    # Analyze attention weights
    print("\nAnalyzing attention weights...")
    attention_path = './test_gaa_outputs/attention_weights_epoch_0.pkl'
    analyze_attention_weights(attention_path, ['mRNA', 'miRNA', 'Methylation'])
    
    # Plot clustering performance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(clustering_df['k'], clustering_df['ari'], 'bo-', label='ARI')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Adjusted Rand Index')
    plt.title('Clustering Performance (ARI)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(clustering_df['k'], clustering_df['silhouette'], 'ro-', label='Silhouette')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Performance (Silhouette)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(demo_output_dir, 'clustering_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDemo completed! Results saved to: {demo_output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(demo_output_dir):
        print(f"  - {file}")


if __name__ == "__main__":
    demonstrate_gaa_usage()