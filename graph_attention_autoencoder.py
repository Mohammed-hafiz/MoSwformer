"""
Graph Attention Autoencoder for Multi-Omics Data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from typing import List, Dict, Tuple, Optional
import numpy as np
from utils.graph_utils import create_multimodal_graph, build_feature_graph


class GraphAttentionEncoder(nn.Module):
    """Graph Attention Encoder for single omics modality"""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.2,
                 graph_type: str = 'feature'):
        super(GraphAttentionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.graph_type = graph_type
        
        # Input projection - different based on graph type
        if graph_type == 'feature':
            # For feature graphs, each node is a feature, input is sample values
            proj_input_dim = 1
        else:
            # For sample graphs, each node is a sample, input is feature vector
            proj_input_dim = input_dim
            
        self.input_projection = nn.Linear(proj_input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(proj_input_dim, hidden_dim // num_heads, heads=num_heads, 
                   dropout=dropout, concat=True)
        )
        
        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                       dropout=dropout, concat=True)
            )
        
        # Final GAT layer (no concatenation)
        if num_layers > 1:
            self.gat_layers.append(
                GATConv(hidden_dim, latent_dim, heads=1, dropout=dropout, concat=False)
            )
        else:
            # Single layer case - output directly to latent_dim
            self.gat_layers[0] = GATConv(proj_input_dim, latent_dim, heads=1, dropout=dropout, concat=False)
        
        # Normalization layers - match the GAT output dimensions
        self.norm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:  # Last layer
                norm_dim = latent_dim
            else:  # Intermediate layers
                norm_dim = hidden_dim
            self.norm_layers.append(nn.LayerNorm(norm_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling for sample-level representation
        self.global_pool_mean = global_mean_pool
        self.global_pool_max = global_max_pool
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None, 
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GAT encoder
        
        Args:
            x: Node features of shape (num_nodes, 1) - each feature is a node
            edge_index: Graph edge indices
            edge_weight: Edge weights
            batch: Batch assignment for nodes
            
        Returns:
            Dictionary containing node embeddings, attention weights, and global representation
        """
        # Store attention weights
        attention_weights = []
        
        # Start with input features - no projection needed, GAT handles it
        h = x
        
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT
            h_new = gat_layer(h, edge_index, edge_weight)
            
            # Apply normalization and dropout
            h = self.norm_layers[i](h_new)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Global pooling for sample representation
        if batch is not None:
            global_mean = self.global_pool_mean(h, batch)
            global_max = self.global_pool_max(h, batch)
            global_repr = torch.cat([global_mean, global_max], dim=1)
        else:
            # Single sample case
            global_repr = torch.cat([h.mean(dim=0, keepdim=True), 
                                   h.max(dim=0, keepdim=True)[0]], dim=1)
        
        return {
            'node_embeddings': h,
            'global_repr': global_repr,
            'attention_weights': attention_weights
        }

    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor, 
                             edge_weight: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get attention weights for visualization
        
        Args:
            x: Node features
            edge_index: Graph edge indices  
            edge_weight: Edge weights
            
        Returns:
            List of attention weight tensors
        """
        attention_weights = []
        h = x
        
        for i, gat_layer in enumerate(self.gat_layers):
            h_new, (edge_index_with_attn, attn_weights) = gat_layer(h, edge_index, edge_weight, return_attention_weights=True)
            attention_weights.append((edge_index_with_attn, attn_weights))
            
            h = self.norm_layers[i](h_new)
            h = F.relu(h)
            h = self.dropout(h)
        
        return attention_weights


class MultiModalFusion(nn.Module):
    """Fusion module for combining multi-modal latent representations"""
    
    def __init__(self, latent_dims: List[int], fused_dim: int, 
                 fusion_type: str = 'attention'):
        super(MultiModalFusion, self).__init__()
        
        self.latent_dims = latent_dims
        self.fused_dim = fused_dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion_layer = nn.Linear(sum(latent_dims) * 2, fused_dim)  # *2 for mean+max pooling
            
        elif fusion_type == 'attention':
            # Project each modality to same dimension
            self.modality_projections = nn.ModuleList([
                nn.Linear(dim * 2, fused_dim) for dim in latent_dims
            ])
            
            # Attention mechanism
            self.attention = nn.MultiheadAttention(fused_dim, num_heads=4, batch_first=True)
            
        elif fusion_type == 'weighted':
            # Learnable weights for each modality
            self.modality_weights = nn.Parameter(torch.ones(len(latent_dims)))
            self.fusion_layer = nn.Linear(sum(latent_dims) * 2, fused_dim)
        
    def forward(self, modality_reprs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-modal representations
        
        Args:
            modality_reprs: List of latent representations from each modality
            
        Returns:
            Fused representation
        """
        if self.fusion_type == 'concat':
            fused = torch.cat(modality_reprs, dim=1)
            return self.fusion_layer(fused)
            
        elif self.fusion_type == 'attention':
            # Project each modality
            projected = []
            for i, repr in enumerate(modality_reprs):
                proj = self.modality_projections[i](repr)
                projected.append(proj)
            
            # Stack for attention
            modality_stack = torch.stack(projected, dim=1)  # (batch, n_modalities, fused_dim)
            
            # Self-attention across modalities
            attn_out, _ = self.attention(modality_stack, modality_stack, modality_stack)
            
            # Global pooling across modalities
            fused = attn_out.mean(dim=1)
            return fused
            
        elif self.fusion_type == 'weighted':
            # Weighted combination
            weights = F.softmax(self.modality_weights, dim=0)
            weighted_reprs = []
            for i, repr in enumerate(modality_reprs):
                weighted_reprs.append(weights[i] * repr)
            
            fused = torch.cat(weighted_reprs, dim=1)
            return self.fusion_layer(fused)


class GraphAttentionDecoder(nn.Module):
    """Graph Attention Decoder for reconstructing omics data"""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super(GraphAttentionDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Decoder layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruct omics data
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed omics data
        """
        return self.decoder(z)


class GraphAttentionAutoencoder(nn.Module):
    """Complete Graph Attention Autoencoder for Multi-Omics Data"""
    
    def __init__(self, input_dims: List[int], latent_dim: int = 64, 
                 hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, fusion_type: str = 'attention',
                 graph_config: Dict = None):
        super(GraphAttentionAutoencoder, self).__init__()
        
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.num_modalities = len(input_dims)
        
        # Graph configuration
        self.graph_config = graph_config or {
            'k_neighbors': 10,
            'graph_type': 'feature',
            'similarity_method': 'cosine'
        }
        
        # Encoders for each modality
        self.encoders = nn.ModuleList([
            GraphAttentionEncoder(input_dim, hidden_dim, latent_dim, 
                                num_heads, num_layers, dropout, self.graph_config['graph_type'])
            for input_dim in input_dims
        ])
        
        # Fusion module
        self.fusion = MultiModalFusion([latent_dim] * self.num_modalities, 
                                     latent_dim, fusion_type)
        
        # Decoders for each modality
        self.decoders = nn.ModuleList([
            GraphAttentionDecoder(latent_dim, hidden_dim, input_dim, 
                                num_layers, dropout)
            for input_dim in input_dims
        ])
        
    def encode(self, data_list: List[torch.Tensor], 
               graphs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Encode multi-omics data to latent representations
        
        Args:
            data_list: List of omics data tensors
            graphs: Pre-computed graphs (optional)
            
        Returns:
            Dictionary containing modality-specific and fused latent representations
        """
        if graphs is None:
            # Build graphs on-the-fly
            graphs = create_multimodal_graph(
                data_list, 
                k=self.graph_config['k_neighbors'],
                graph_type=self.graph_config['graph_type'],
                method=self.graph_config['similarity_method']
            )
        
        modality_outputs = {}
        modality_reprs = []
        all_attention_weights = []
        
        for i, (data, encoder) in enumerate(zip(data_list, self.encoders)):
            graph_key = f'modality_{i}'
            edge_index = graphs[graph_key]['edge_index']
            edge_weight = graphs[graph_key]['edge_weight']
            
            # Prepare node features (each feature as a node)
            if self.graph_config['graph_type'] == 'feature':
                # Shape: (batch_size, n_features) -> (n_features, batch_size)
                node_features = data.t()  # (n_features, batch_size)
                # For single sample case, add feature dimension
                if node_features.dim() == 1:
                    node_features = node_features.unsqueeze(-1)  # (n_features, 1)
            else:
                # For sample graphs, each node is a sample
                node_features = data  # (batch_size, n_features)
                if node_features.dim() == 1:
                    node_features = node_features.unsqueeze(0)  # (1, n_features)
            
            # Encode
            output = encoder(node_features, edge_index, edge_weight)
            
            modality_outputs[f'modality_{i}'] = output
            modality_reprs.append(output['global_repr'])
            all_attention_weights.extend(output['attention_weights'])
        
        # Fuse modality representations
        fused_repr = self.fusion(modality_reprs)
        
        return {
            'modality_outputs': modality_outputs,
            'modality_reprs': modality_reprs,
            'fused_repr': fused_repr,
            'attention_weights': all_attention_weights
        }
    
    def decode(self, fused_repr: torch.Tensor) -> List[torch.Tensor]:
        """
        Decode fused representation to reconstruct all modalities
        
        Args:
            fused_repr: Fused latent representation
            
        Returns:
            List of reconstructed omics data
        """
        reconstructions = []
        for decoder in self.decoders:
            recon = decoder(fused_repr)
            reconstructions.append(recon)
        
        return reconstructions
    
    def forward(self, data_list: List[torch.Tensor], 
                graphs: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass: encode -> decode
        
        Args:
            data_list: List of omics data tensors
            graphs: Pre-computed graphs
            
        Returns:
            Dictionary containing all outputs
        """
        # Encode
        encoding_output = self.encode(data_list, graphs)
        
        # Decode
        reconstructions = self.decode(encoding_output['fused_repr'])
        
        return {
            **encoding_output,
            'reconstructions': reconstructions
        }
    
    def get_latent_representation(self, data_list: List[torch.Tensor],
                                graphs: Optional[Dict] = None) -> torch.Tensor:
        """
        Get fused latent representation for downstream tasks
        
        Args:
            data_list: List of omics data tensors
            graphs: Pre-computed graphs
            
        Returns:
            Fused latent representation
        """
        encoding_output = self.encode(data_list, graphs)
        return encoding_output['fused_repr']
    
    def get_attention_weights(self, data_list: List[torch.Tensor],
                            graphs: Optional[Dict] = None) -> List[List[torch.Tensor]]:
        """
        Get attention weights for visualization
        
        Args:
            data_list: List of omics data tensors
            graphs: Pre-computed graphs
            
        Returns:
            List of attention weight tensors for each modality
        """
        if graphs is None:
            graphs = create_multimodal_graph(
                data_list, 
                k=self.graph_config['k_neighbors'],
                graph_type=self.graph_config['graph_type'],
                method=self.graph_config['similarity_method']
            )
        
        all_attention_weights = []
        
        for i, (data, encoder) in enumerate(zip(data_list, self.encoders)):
            graph_key = f'modality_{i}'
            edge_index = graphs[graph_key]['edge_index']
            edge_weight = graphs[graph_key]['edge_weight']
            
            # Prepare node features
            if self.graph_config['graph_type'] == 'feature':
                node_features = data.t()
                if node_features.dim() == 1:
                    node_features = node_features.unsqueeze(-1)
            else:
                node_features = data
                if node_features.dim() == 1:
                    node_features = node_features.unsqueeze(0)
            
            # Get attention weights
            attn_weights = encoder.get_attention_weights(node_features, edge_index, edge_weight)
            all_attention_weights.append(attn_weights)
        
        return all_attention_weights