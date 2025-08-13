"""
Graph construction utilities for multi-omics data
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import torch_geometric.utils
from typing import Tuple, Optional


def compute_similarity_matrix(data: torch.Tensor, method: str = 'cosine') -> torch.Tensor:
    """
    Compute similarity matrix between samples or features.
    
    Args:
        data: Input tensor of shape (n_samples, n_features)
        method: Similarity metric ('cosine', 'correlation', 'euclidean')
    
    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    if method == 'cosine':
        # Normalize data for cosine similarity
        data_norm = F.normalize(data, p=2, dim=1)
        similarity = torch.mm(data_norm, data_norm.t())
    
    elif method == 'correlation':
        # Pearson correlation
        data_centered = data - data.mean(dim=1, keepdim=True)
        data_norm = F.normalize(data_centered, p=2, dim=1)
        similarity = torch.mm(data_norm, data_norm.t())
    
    elif method == 'euclidean':
        # Convert euclidean distance to similarity
        distances = torch.cdist(data, data, p=2)
        # Use RBF kernel to convert distance to similarity
        sigma = distances.std()
        similarity = torch.exp(-distances.pow(2) / (2 * sigma.pow(2)))
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity


def build_knn_graph(similarity_matrix: torch.Tensor, k: int = 10, 
                   include_self: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-nearest neighbor graph from similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix of shape (n_nodes, n_nodes)
        k: Number of nearest neighbors
        include_self: Whether to include self-connections
    
    Returns:
        edge_index: Edge indices of shape (2, num_edges)
        edge_weight: Edge weights of shape (num_edges,)
    """
    n_nodes = similarity_matrix.size(0)
    
    if not include_self:
        # Remove self-connections by setting diagonal to -inf
        similarity_matrix = similarity_matrix.clone()
        similarity_matrix.fill_diagonal_(-float('inf'))
    
    # Get top-k neighbors for each node
    _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
    
    # Create edge list
    source_nodes = torch.arange(n_nodes).repeat_interleave(k)
    target_nodes = top_k_indices.flatten()
    
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    # Get corresponding edge weights
    edge_weight = similarity_matrix[source_nodes, target_nodes]
    
    # Make graph undirected by adding reverse edges
    reverse_edge_index = torch.stack([target_nodes, source_nodes], dim=0)
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    
    # Remove duplicate edges and average weights
    edge_index, edge_weight = torch_geometric.utils.coalesce(
        edge_index, edge_weight, num_nodes=n_nodes, reduce='mean'
    )
    
    return edge_index, edge_weight


def build_feature_graph(data: torch.Tensor, k: int = 10, 
                       method: str = 'cosine') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph connecting similar features.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        k: Number of nearest neighbors
        method: Similarity method
    
    Returns:
        edge_index: Edge indices connecting features
        edge_weight: Edge weights
    """
    # Transpose to get features as rows
    feature_data = data.t()  # Shape: (n_features, n_samples)
    
    # Compute feature similarity
    similarity_matrix = compute_similarity_matrix(feature_data, method)
    
    # Build k-NN graph
    edge_index, edge_weight = build_knn_graph(similarity_matrix, k, include_self=False)
    
    return edge_index, edge_weight


def build_sample_graph(data: torch.Tensor, k: int = 10, 
                      method: str = 'cosine') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph connecting similar samples.
    
    Args:
        data: Input data of shape (n_samples, n_features)  
        k: Number of nearest neighbors
        method: Similarity method
    
    Returns:
        edge_index: Edge indices connecting samples
        edge_weight: Edge weights
    """
    # Compute sample similarity
    similarity_matrix = compute_similarity_matrix(data, method)
    
    # Build k-NN graph
    edge_index, edge_weight = build_knn_graph(similarity_matrix, k, include_self=False)
    
    return edge_index, edge_weight


def create_multimodal_graph(data_list: list, k: int = 10, 
                           graph_type: str = 'feature',
                           method: str = 'cosine') -> dict:
    """
    Create graphs for multiple omics modalities.
    
    Args:
        data_list: List of data tensors for each modality
        k: Number of nearest neighbors
        graph_type: 'feature' or 'sample' graph
        method: Similarity method
    
    Returns:
        Dictionary containing graphs for each modality
    """
    graphs = {}
    
    for i, data in enumerate(data_list):
        if graph_type == 'feature':
            edge_index, edge_weight = build_feature_graph(data, k, method)
        elif graph_type == 'sample':
            edge_index, edge_weight = build_sample_graph(data, k, method)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        graphs[f'modality_{i}'] = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'num_nodes': data.size(1) if graph_type == 'feature' else data.size(0)
        }
    
    return graphs


def add_self_loops(edge_index: torch.Tensor, edge_weight: torch.Tensor, 
                   num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add self-loops to graph.
    
    Args:
        edge_index: Edge indices
        edge_weight: Edge weights
        num_nodes: Number of nodes
    
    Returns:
        Updated edge_index and edge_weight with self-loops
    """
    # Create self-loop edges
    self_loop_index = torch.arange(num_nodes).repeat(2, 1)
    self_loop_weight = torch.ones(num_nodes)
    
    # Concatenate with existing edges
    edge_index = torch.cat([edge_index, self_loop_index], dim=1)
    edge_weight = torch.cat([edge_weight, self_loop_weight], dim=0)
    
    return edge_index, edge_weight