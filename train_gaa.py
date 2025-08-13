"""
Training script for Graph Attention Autoencoder
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataloader import data_set  
from graph_attention_autoencoder import GraphAttentionAutoencoder
from config.gaa_config import GAAConfig
from utils.graph_utils import create_multimodal_graph


def reconstruction_loss(reconstructions: list, targets: list, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute reconstruction loss for multi-modal data
    
    Args:
        reconstructions: List of reconstructed data tensors
        targets: List of target data tensors
        reduction: Loss reduction method
        
    Returns:
        Total reconstruction loss
    """
    total_loss = 0
    for recon, target in zip(reconstructions, targets):
        loss = nn.MSELoss(reduction=reduction)(recon, target)
        total_loss += loss
    
    return total_loss


def latent_regularization_loss(latent_repr: torch.Tensor, reg_type: str = 'l2') -> torch.Tensor:
    """
    Regularization loss for latent representations
    
    Args:
        latent_repr: Latent representation tensor
        reg_type: Type of regularization ('l1' or 'l2')
        
    Returns:
        Regularization loss
    """
    if reg_type == 'l1':
        return torch.mean(torch.abs(latent_repr))
    elif reg_type == 'l2':
        return torch.mean(latent_repr.pow(2))
    else:
        return torch.tensor(0.0)


def attention_regularization_loss(attention_weights: list) -> torch.Tensor:
    """
    Regularization loss for attention weights to encourage diversity
    
    Args:
        attention_weights: List of attention weight tensors
        
    Returns:
        Attention regularization loss
    """
    if not attention_weights:
        return torch.tensor(0.0)
    
    total_loss = 0
    for attn in attention_weights:
        # Encourage attention diversity (entropy-based)
        if len(attn) == 2:  # (edge_index, attention_values)
            attn_values = attn[1]
            # Normalize attention values
            attn_normalized = torch.softmax(attn_values, dim=0)
            # Entropy loss (negative entropy to encourage diversity)
            entropy = -torch.sum(attn_normalized * torch.log(attn_normalized + 1e-8))
            total_loss += -entropy  # Maximize entropy
    
    return total_loss / len(attention_weights) if attention_weights else torch.tensor(0.0)


def evaluate_model(model: GraphAttentionAutoencoder, data_loader: DataLoader, 
                  device: str, config: GAAConfig) -> dict:
    """
    Evaluate model performance
    
    Args:
        model: GAA model
        data_loader: Data loader
        device: Device to use
        config: Configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_samples = 0
    
    all_reconstructions = [[] for _ in range(3)]  # 3 modalities
    all_targets = [[] for _ in range(3)]
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data1 = sample['data1'].to(device)
            data2 = sample['data2'].to(device) 
            data3 = sample['data3'].to(device)
            
            data_list = [data1, data2, data3]
            batch_size = data1.size(0)
            
            # Forward pass
            for i in range(batch_size):
                single_data = [data[i:i+1] for data in data_list]
                
                output = model(single_data)
                reconstructions = output['reconstructions']
                
                # Compute losses
                recon_loss = reconstruction_loss(reconstructions, single_data)
                
                total_recon_loss += recon_loss.item()
                total_loss += recon_loss.item()
                total_samples += 1
                
                # Store for detailed evaluation
                for j, (recon, target) in enumerate(zip(reconstructions, single_data)):
                    all_reconstructions[j].append(recon.cpu().numpy())
                    all_targets[j].append(target.cpu().numpy())
    
    # Compute detailed metrics
    metrics = {
        'total_loss': total_loss / total_samples,
        'reconstruction_loss': total_recon_loss / total_samples
    }
    
    # Per-modality metrics
    for i in range(3):
        recons = np.concatenate(all_reconstructions[i], axis=0)
        targets = np.concatenate(all_targets[i], axis=0)
        
        mse = mean_squared_error(targets.flatten(), recons.flatten())
        mae = mean_absolute_error(targets.flatten(), recons.flatten())
        
        metrics[f'modality_{i}_mse'] = mse
        metrics[f'modality_{i}_mae'] = mae
    
    return metrics


def save_results(model: GraphAttentionAutoencoder, data_loader: DataLoader,
                device: str, config: GAAConfig, epoch: int):
    """
    Save model results including latent representations and attention weights
    
    Args:
        model: GAA model
        data_loader: Data loader
        device: Device
        config: Configuration
        epoch: Current epoch
    """
    model.eval()
    
    latent_representations = []
    attention_weights_all = []
    sample_ids = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data1 = sample['data1'].to(device)
            data2 = sample['data2'].to(device)
            data3 = sample['data3'].to(device)
            
            data_list = [data1, data2, data3]
            batch_size = data1.size(0)
            
            for i in range(batch_size):
                single_data = [data[i:i+1] for data in data_list]
                
                # Get latent representation
                latent = model.get_latent_representation(single_data)
                latent_representations.append(latent.cpu().numpy())
                
                # Get attention weights
                if config.save_attention:
                    attn_weights = model.get_attention_weights(single_data)
                    attention_weights_all.append(attn_weights)
                
                sample_ids.append(f'batch_{batch_idx}_sample_{i}')
    
    # Save latent representations
    if config.save_latent:
        latent_array = np.concatenate(latent_representations, axis=0)
        latent_df = pd.DataFrame(latent_array, index=sample_ids)
        latent_path = os.path.join(config.output_dir, f'latent_representations_epoch_{epoch}.csv')
        latent_df.to_csv(latent_path)
    
    # Save attention weights
    if config.save_attention and attention_weights_all:
        attn_path = os.path.join(config.output_dir, f'attention_weights_epoch_{epoch}.pkl')
        with open(attn_path, 'wb') as f:
            pickle.dump(attention_weights_all, f)


def train_gaa(config: GAAConfig):
    """
    Main training function for Graph Attention Autoencoder
    
    Args:
        config: GAA configuration
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config.get_device())
    print(f"Using device: {device}")
    
    # Load data
    train_dataset = data_set(config.data_folder, train_flag=True)
    test_dataset = data_set(config.data_folder, train_flag=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get data dimensions from a sample
    sample = train_dataset[0]
    input_dims = [sample['data1'].shape[0], sample['data2'].shape[0], sample['data3'].shape[0]]
    print(f"Input dimensions: {input_dims}")
    
    # Initialize model
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
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), 
                          lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # Setup scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training history
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    early_stop_counter = 0
    
    print("Starting training...")
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_latent_reg = 0
        epoch_attn_reg = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch_idx, sample in enumerate(pbar):
            data1 = sample['data1'].to(device)
            data2 = sample['data2'].to(device)
            data3 = sample['data3'].to(device)
            
            data_list = [data1, data2, data3]
            batch_size = data1.size(0)
            batch_loss = 0
            batch_recon_loss = 0
            batch_latent_reg = 0
            batch_attn_reg = 0
            
            optimizer.zero_grad()
            
            # Process each sample individually (due to different graph structures)
            for i in range(batch_size):
                single_data = [data[i:i+1] for data in data_list]
                
                # Forward pass
                output = model(single_data)
                reconstructions = output['reconstructions']
                fused_repr = output['fused_repr']
                attention_weights = output['attention_weights']
                
                # Compute losses
                recon_loss = reconstruction_loss(reconstructions, single_data)
                latent_reg = latent_regularization_loss(fused_repr)
                attn_reg = attention_regularization_loss(attention_weights)
                
                # Total loss
                total_loss = (config.reconstruction_weight * recon_loss + 
                            config.latent_reg_weight * latent_reg +
                            config.attention_reg_weight * attn_reg)
                
                total_loss.backward()
                
                batch_loss += total_loss.item()
                batch_recon_loss += recon_loss.item()
                batch_latent_reg += latent_reg.item()
                batch_attn_reg += attn_reg.item()
            
            optimizer.step()
            
            # Update metrics
            epoch_train_loss += batch_loss
            epoch_recon_loss += batch_recon_loss
            epoch_latent_reg += batch_latent_reg
            epoch_attn_reg += batch_attn_reg
            num_batches += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{batch_loss/batch_size:.4f}',
                'Recon': f'{batch_recon_loss/batch_size:.4f}'
            })
        
        # Average losses
        avg_train_loss = epoch_train_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_latent_reg = epoch_latent_reg / num_batches
        avg_attn_reg = epoch_attn_reg / num_batches
        
        train_losses.append(avg_train_loss)
        
        # Evaluation phase
        test_metrics = evaluate_model(model, test_loader, device, config)
        test_loss = test_metrics['total_loss']
        test_losses.append(test_loss)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{config.num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f} (Recon: {avg_recon_loss:.6f}, '
              f'Latent: {avg_latent_reg:.6f}, Attn: {avg_attn_reg:.6f})')
        print(f'  Test Loss: {test_loss:.6f}')
        
        for i in range(3):
            print(f'  Modality {i} - MSE: {test_metrics[f"modality_{i}_mse"]:.6f}, '
                  f'MAE: {test_metrics[f"modality_{i}_mae"]:.6f}')
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 
                      os.path.join(config.output_dir, 'best_gaa_model.pth'))
            
            # Save results for best model
            save_results(model, test_loader, device, config, epoch)
            
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= config.early_stopping:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break
        
        # Save periodic results
        if (epoch + 1) % 10 == 0:
            save_results(model, test_loader, device, config, epoch)
    
    # Save final model and training history
    torch.save(model.state_dict(), 
              os.path.join(config.output_dir, 'final_gaa_model.pth'))
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'test_loss': test_losses[:len(train_losses)]
    })
    history_df.to_csv(os.path.join(config.output_dir, 'training_history.csv'), index=False)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(test_losses[:len(train_losses)], label='Test Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAA Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\nTraining completed!')
    print(f'Best test loss: {best_loss:.6f}')
    print(f'Results saved to: {config.output_dir}')


if __name__ == "__main__":
    # Default configuration
    config = GAAConfig(
        data_folder='./BRCA_split/BRCA',
        batch_size=16,  # Smaller batch size due to individual graph processing
        latent_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        k_neighbors=10,
        graph_type='feature',
        similarity_method='cosine',
        learning_rate=0.001,
        weight_decay=1e-5,
        num_epochs=100,
        early_stopping=20,
        reconstruction_weight=1.0,
        latent_reg_weight=0.01,
        attention_reg_weight=0.001,
        output_dir='./gaa_outputs'
    )
    
    print("Graph Attention Autoencoder Configuration:")
    print(config)
    print("\n" + "="*50 + "\n")
    
    # Start training
    train_gaa(config)