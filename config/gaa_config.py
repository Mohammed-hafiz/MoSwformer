"""
Configuration file for Graph Attention Autoencoder
"""
import argparse
import torch


def gaa_config_parser():
    """Configuration parser for Graph Attention Autoencoder"""
    
    parser = argparse.ArgumentParser(description="Graph Attention Autoencoder Configuration")
    
    # Data parameters
    parser.add_argument('--data_folder', type=str, default='./BRCA_split/BRCA',
                       help='Path to data folder')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    # Model architecture parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension for each modality')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension in GAT layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads in GAT')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GAT layers per modality')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Graph construction parameters  
    parser.add_argument('--graph_type', type=str, default='feature',
                       choices=['feature', 'sample'],
                       help='Type of graph to construct')
    parser.add_argument('--k_neighbors', type=int, default=10,
                       help='Number of k-nearest neighbors for graph construction')
    parser.add_argument('--similarity_method', type=str, default='cosine',
                       choices=['cosine', 'correlation', 'euclidean'],
                       help='Similarity method for graph construction')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience')
    
    # Loss function parameters
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--latent_reg_weight', type=float, default=0.01,
                       help='Weight for latent regularization')
    parser.add_argument('--attention_reg_weight', type=float, default=0.001,
                       help='Weight for attention regularization')
    
    # Visualization and output parameters
    parser.add_argument('--save_attention', type=bool, default=True,
                       help='Whether to save attention weights')
    parser.add_argument('--save_latent', type=bool, default=True,
                       help='Whether to save latent representations')
    parser.add_argument('--output_dir', type=str, default='./gaa_outputs',
                       help='Output directory for results')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    return parser


class GAAConfig:
    """Configuration class for Graph Attention Autoencoder"""
    
    def __init__(self, **kwargs):
        # Set default values
        self.data_folder = './BRCA_split/BRCA'
        self.batch_size = 32
        
        # Model parameters
        self.latent_dim = 64
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.2
        
        # Graph parameters
        self.graph_type = 'feature'
        self.k_neighbors = 10
        self.similarity_method = 'cosine'
        
        # Training parameters
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.num_epochs = 100
        self.early_stopping = 20
        
        # Loss weights
        self.reconstruction_weight = 1.0
        self.latent_reg_weight = 0.01
        self.attention_reg_weight = 0.001
        
        # Output parameters
        self.save_attention = True
        self.save_latent = True
        self.output_dir = './gaa_outputs'
        
        # Device
        self.device = 'auto'
        
        # Update with provided arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_device(self):
        """Get the appropriate device"""
        if self.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"GAAConfig({', '.join(attrs)})"


if __name__ == "__main__":
    # Example usage
    parser = gaa_config_parser()
    args = parser.parse_args()
    
    config = GAAConfig(**vars(args))
    print(config)