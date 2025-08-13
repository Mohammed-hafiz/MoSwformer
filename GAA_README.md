# Graph Attention Autoencoder for Multi-Omics Data

This implementation provides a Graph Attention Autoencoder (GAA) for extracting meaningful latent representations from multi-omics data. The model processes three different omics data modalities and learns their relationships using graph attention mechanisms.

## Features

- **Multi-modal Graph Attention Encoder**: Separate GAT encoders for each omics type
- **Latent Fusion Layer**: Combines latent representations from all three modalities using attention
- **Multi-modal Decoder**: Reconstructs all three omics data from fused latent space
- **Attention Weight Analysis**: Provides attention weight visualization capabilities
- **Flexible Graph Construction**: Supports different similarity metrics and graph types

## Model Architecture

```
Input: [Omics1, Omics2, Omics3] → Graph Construction → GAT Encoders → 
Fusion Layer → Fused Latent Representation → Decoders → 
Reconstructed [Omics1, Omics2, Omics3]
```

### Key Components

1. **Graph Construction** (`utils/graph_utils.py`):
   - Feature-based or sample-based similarity graphs
   - k-NN graph construction with various similarity metrics
   - Support for cosine similarity, Pearson correlation, and Euclidean distance

2. **Graph Attention Encoder** (`graph_attention_autoencoder.py`):
   - Multi-head graph attention layers
   - Layer normalization and dropout
   - Global pooling for sample-level representations

3. **Multi-Modal Fusion**:
   - Attention-based fusion of modality-specific representations
   - Learnable combination weights
   - Configurable fusion strategies

4. **Decoder Network**:
   - Multi-layer perceptron for reconstruction
   - Separate decoders for each omics modality

## Installation

Required dependencies:
```bash
pip install torch torch-geometric scikit-learn pandas matplotlib numpy
```

## Usage

### Basic Training

```python
from train_gaa import train_gaa
from config.gaa_config import GAAConfig

# Configure the model
config = GAAConfig(
    data_folder='./BRCA_split/BRCA',
    batch_size=16,
    latent_dim=64,
    hidden_dim=128,
    num_heads=4,
    num_layers=2,
    k_neighbors=10,
    graph_type='feature',
    similarity_method='cosine',
    num_epochs=100,
    learning_rate=0.001
)

# Start training
train_gaa(config)
```

### Using Trained Model for Feature Extraction

```python
from graph_attention_autoencoder import GraphAttentionAutoencoder
from dataloader import data_set
import torch

# Load data
dataset = data_set('./BRCA_split/BRCA', train_flag=False)
sample = dataset[0]

# Prepare input
data_list = [
    sample['data1'].unsqueeze(0),
    sample['data2'].unsqueeze(0), 
    sample['data3'].unsqueeze(0)
]

# Load trained model
model = GraphAttentionAutoencoder(...)
model.load_state_dict(torch.load('path/to/model.pth'))

# Extract latent representation
latent_features = model.get_latent_representation(data_list)
print(f"Latent features shape: {latent_features.shape}")
```

### Complete Demo

Run the demonstration script to see the full pipeline:

```bash
python gaa_demo.py
```

This will:
- Load a trained model
- Extract latent features from test data
- Visualize the latent space using PCA
- Perform clustering analysis
- Analyze attention weights

## Configuration Options

### Model Parameters
- `latent_dim`: Dimension of latent representation (default: 64)
- `hidden_dim`: Hidden dimension in GAT layers (default: 128)
- `num_heads`: Number of attention heads (default: 4)
- `num_layers`: Number of GAT layers (default: 2)
- `dropout`: Dropout rate (default: 0.2)

### Graph Construction Parameters
- `graph_type`: 'feature' or 'sample' (default: 'feature')
- `k_neighbors`: Number of k-nearest neighbors (default: 10)
- `similarity_method`: 'cosine', 'correlation', or 'euclidean' (default: 'cosine')

### Training Parameters
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: Weight decay (default: 1e-5)
- `num_epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 16)
- `early_stopping`: Early stopping patience (default: 20)

### Loss Function Weights
- `reconstruction_weight`: Weight for reconstruction loss (default: 1.0)
- `latent_reg_weight`: Weight for latent regularization (default: 0.01)
- `attention_reg_weight`: Weight for attention regularization (default: 0.001)

## Input Data Format

The model expects data in the format provided by the existing `dataloader.py`:

- **data1**: First omics modality (e.g., mRNA expression)
- **data2**: Second omics modality (e.g., miRNA expression)  
- **data3**: Third omics modality (e.g., DNA methylation)
- **label**: Sample labels (for evaluation)

Data dimensions in the BRCA dataset:
- Modality 1: 1000 features
- Modality 2: 1000 features
- Modality 3: 503 features

## Outputs

### Training Outputs
- `best_gaa_model.pth`: Best model weights
- `final_gaa_model.pth`: Final model weights
- `training_history.csv`: Training and validation loss history
- `training_curves.png`: Training curves plot
- `latent_representations_epoch_X.csv`: Latent representations for each epoch
- `attention_weights_epoch_X.pkl`: Attention weights for visualization

### Analysis Outputs
- `latent_space_pca.png`: PCA visualization of latent space
- `clustering_results.csv`: Clustering performance metrics
- `clustering_performance.png`: Clustering performance plots
- `latent_features_analysis.csv`: Latent features with clustering results

## File Structure

```
MoSwformer/
├── graph_attention_autoencoder.py  # Main GAA model
├── train_gaa.py                   # Training script
├── gaa_demo.py                    # Demonstration script
├── config/
│   ├── __init__.py
│   └── gaa_config.py              # Configuration classes
├── utils/
│   ├── __init__.py
│   └── graph_utils.py             # Graph construction utilities
└── dataloader.py                  # Existing data loader
```

## Performance Considerations

- **Memory Usage**: Graph construction can be memory-intensive for large datasets
- **Batch Processing**: Individual samples are processed separately due to different graph structures
- **GPU Support**: Model supports CUDA acceleration
- **Scalability**: Consider reducing `k_neighbors` for very large datasets

## Integration with Existing Code

The GAA implementation is designed to be compatible with the existing MoSwformer codebase:

- Uses the same `data_set` class from `dataloader.py`
- Maintains the same data format and file organization
- Compatible with the BRCA dataset structure
- Can be used alongside existing models

## Advanced Usage

### Custom Graph Construction

```python
from utils.graph_utils import create_multimodal_graph

# Build custom graphs
graphs = create_multimodal_graph(
    data_list, 
    k=15, 
    graph_type='sample',
    method='correlation'
)
```

### Attention Weight Visualization

```python
# Get attention weights for interpretability
attention_weights = model.get_attention_weights(data_list)

# Process attention weights for visualization
for modality_idx, modality_attn in enumerate(attention_weights):
    for layer_idx, layer_attn in enumerate(modality_attn):
        edge_index, attn_values = layer_attn
        # Visualize attention patterns
```

### Custom Loss Functions

```python
# Modify train_gaa.py to add custom regularization
def custom_loss(output, targets, model):
    recon_loss = reconstruction_loss(output['reconstructions'], targets)
    custom_reg = your_custom_regularization(model)
    return recon_loss + 0.01 * custom_reg
```

## Citation

If you use this Graph Attention Autoencoder in your research, please cite:

```
Graph Attention Autoencoder for Multi-Omics Data Analysis
[Add appropriate citation information]
```

## License

This implementation follows the same license as the parent MoSwformer repository.