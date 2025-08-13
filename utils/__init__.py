"""
Utilities package for Graph Attention Autoencoder
"""

from .graph_utils import (
    compute_similarity_matrix,
    build_knn_graph,
    build_feature_graph,
    build_sample_graph,
    create_multimodal_graph,
    add_self_loops
)

__all__ = [
    'compute_similarity_matrix',
    'build_knn_graph', 
    'build_feature_graph',
    'build_sample_graph',
    'create_multimodal_graph',
    'add_self_loops'
]