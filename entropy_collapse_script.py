import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def get_batch_entropy(logits):
    """
    Calculates H_t as described in Step 2: Predictive Entropy Collapse.
    
    Args:
        logits: Raw model outputs (before softmax)
    
    Returns:
        Average entropy across the batch (the Expectation E_x_t)
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate Shannon Entropy: -sum(p * log(p))
    # Adding a small epsilon (1e-9) prevents log(0) errors
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    
    # Average across the batch (the Expectation E_x_t)
    return entropy.mean()


def train_with_entropy_tracking(model, dataloader, criterion, optimizer, num_epochs):
    """
    Integrated training loop that tracks entropy collapse.
    
    Args:
        model: Neural network model
        dataloader: DataLoader with training data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
    
    Returns:
        Dictionary containing training history
    """
    history = {"entropy": [], "loss": []}
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_entropy = 0.0
        num_batches = 0
        
        for batch in dataloader:
            batch_data, batch_labels = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            # Calculate H_t for this step
            h_t = get_batch_entropy(outputs)
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_entropy += h_t.item()
            num_batches += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Average metrics for the epoch
        avg_loss = epoch_loss / num_batches
        avg_entropy = epoch_entropy / num_batches
        
        history["loss"].append(avg_loss)
        history["entropy"].append(avg_entropy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.4f}")
    
    return history


def plot_entropy_collapse(history):
    """
    Visualize the Epiplexity contribution (ΔH = H_t=0 - H_t=T).
    
    For structured data, we expect a large, smooth collapse in entropy.
    For random data, entropy stays near zero (ΔH ≈ 0).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot entropy over time
    ax1.plot(history["entropy"], linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Predictive Entropy (H_t)")
    ax1.set_title("Entropy Collapse During Training")
    ax1.grid(True, alpha=0.3)
    
    # Calculate delta H
    if len(history["entropy"]) > 0:
        delta_h = history["entropy"][0] - history["entropy"][-1]
        ax1.text(0.05, 0.95, f'ΔH = {delta_h:.4f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot training loss
    ax2.plot(history["loss"], linewidth=2, color='orange')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage with a simple classification task
if __name__ == "__main__":
    # Create a simple model
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, num_classes=3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)  # Return logits (no softmax)
    
    # Create synthetic structured data
    torch.manual_seed(42)
    X_structured = torch.randn(1000, 10)
    y_structured = (X_structured[:, 0] > 0).long()  # Simple rule
    
    dataset = TensorDataset(X_structured, y_structured)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = SimpleClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train and track entropy
    print("Training on structured data:")
    history = train_with_entropy_tracking(model, dataloader, criterion, optimizer, num_epochs=20)
    
    # Visualize results
    plot_entropy_collapse(history)
    
    print(f"\nFinal entropy collapse (ΔH): {history['entropy'][0] - history['entropy'][-1]:.4f}")
    print("For structured data, we expect a significant positive ΔH.")
    print("For random data, ΔH ≈ 0 (entropy stays near zero).")
