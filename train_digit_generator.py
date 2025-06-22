import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
LATENT_DIM = 20
BETA = 1.0  # Beta-VAE parameter for KL divergence weighting

class VAE(nn.Module):
    """
    Variational Autoencoder for handwritten digit generation
    Architecture: Encoder -> Latent Space -> Decoder
    """
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),  # 28*28 = 784 input features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(128, latent_dim)      # Mean of latent distribution
        self.fc_logvar = nn.Linear(128, latent_dim)  # Log variance of latent distribution
        
        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def encode(self, x):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through stochastic nodes"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE Loss Function combining reconstruction loss and KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weighting factor for KL divergence (Beta-VAE)
    
    Returns:
        Total loss, reconstruction loss, KL divergence loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence loss
    # D_KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = reconstruction_loss + beta * kl_divergence
    
    return total_loss, reconstruction_loss, kl_divergence

def train_epoch(model, dataloader, optimizer, epoch, beta=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(-1, 784).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_data, mu, logvar = model(data)
        
        # Calculate loss
        loss, recon_loss, kl_loss = vae_loss_function(recon_data, data, mu, logvar, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.2f}, Recon: {recon_loss.item():.2f}, '
                  f'KL: {kl_loss.item():.2f}')
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def validate_epoch(model, dataloader, beta=1.0):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(-1, 784).to(device)
            recon_data, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss_function(recon_data, data, mu, logvar, beta)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    
    return avg_loss, avg_recon_loss, avg_kl_loss

def generate_samples(model, num_samples=16, latent_dim=20):
    """Generate new samples from the trained model"""
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)
        samples = samples.view(-1, 28, 28)
    return samples

def plot_samples(samples, title="Generated Samples", save_path=None):
    """Plot generated samples"""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title)
    
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i].cpu().numpy(), cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Samples saved to {save_path}")
    plt.show()

def main():
    """Main training function"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize images
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = VAE(latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel Architecture:")
    print(f"Latent Dimension: {LATENT_DIM}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, epoch, BETA)
        
        # Validate
        val_loss, val_recon, val_kl = validate_epoch(model, test_loader, BETA)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        
        # Generate samples every 10 epochs
        if epoch % 10 == 0:
            samples = generate_samples(model, 16, LATENT_DIM)
            plot_samples(samples, f"Generated Samples - Epoch {epoch}", 
                        f"results/samples_epoch_{epoch}.png")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/vae_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_dim': LATENT_DIM,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'latent_dim': LATENT_DIM,
            'beta': BETA
        }
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    
    # Generate final samples
    plt.subplot(1, 2, 2)
    final_samples = generate_samples(model, 16, LATENT_DIM)
    sample_grid = final_samples[:16].cpu().numpy()
    
    # Create 4x4 grid
    grid = np.zeros((4*28, 4*28))
    for i in range(4):
        for j in range(4):
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = sample_grid[i*4 + j]
    
    plt.imshow(grid, cmap='gray')
    plt.title('Final Generated Samples')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/training_results.png')
    plt.show()
    
    print("\nTraining completed!")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main() 