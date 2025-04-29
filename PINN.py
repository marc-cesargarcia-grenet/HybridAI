import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import DataLoader, TensorDataset
import time

# ------------------------------
# Parameters
N = 50
BATCH_SIZE = 1024
EPOCHS = 500
LR = 1e-3
SAVE_PATH_DIR = pathlib.Path(__file__).parent / "models"
DATA_DIR = pathlib.Path(__name__).parent / "data"

#-------------------------------
# Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

# ------------------------------
# PINN Model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.u_net = NeuralNet(input_dim=2, output_dim=1)
        
    def forward(self, x, y):
        inputs = torch.cat((x, y), dim=1)
        return self.u_net(inputs)
    
    def compute_laplacian(self, x, y):
        """Compute the Laplacian of u(x,y) using automatic differentiation"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # Get prediction u(x,y)
        u = self.forward(x, y)
        
        # Compute first derivatives
        grad_u = torch.autograd.grad(
            u, [x, y], 
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(
            grad_u[0], x,
            grad_outputs=torch.ones_like(grad_u[0]),
            create_graph=True
        )[0]
        
        u_yy = torch.autograd.grad(
            grad_u[1], y,
            grad_outputs=torch.ones_like(grad_u[1]),
            create_graph=True
        )[0]
        
        # Laplacian = u_xx + u_yy
        return u_xx + u_yy

# ------------------------------
# Training function
def train_pinn(a, b):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    data = np.load(DATA_DIR / "poisson_data.npz")
    F, U = data['f'], data['u']

    # Find the corresponding FD solution
    a_values = range(10)
    b_values = range(10)
    idx = None
    
    for i, a_val in enumerate(a_values):
        for j, b_val in enumerate(b_values):
            if a_val == a and b_val == b:
                idx = i * len(b_values) + j
                break
        if idx is not None:
            break
    
    if idx is None:
        print(f"Could not find FD solution for a={a}, b={b}")
    
    # Get the FD solution
    u_fd = U[idx].copy()
    f_fd = F[idx].copy()
    
    # Create model
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Generate grid points in the domain (interior and boundary)
    h = 1.0 / (N + 1)
    x_domain = torch.linspace(h, 1-h, N).to(device)
    y_domain = torch.linspace(h, 1-h, N).to(device)
    X, Y = torch.meshgrid(x_domain, y_domain, indexing='ij')
    x_interior = X.reshape(-1, 1)
    y_interior = Y.reshape(-1, 1)
    
    # Create synthetic data for source term f(x,y)
    def source_term(x, y):
        return x * torch.sin(a * np.pi * y) + y * torch.sin(b * np.pi * x)
    
    f_values = source_term(x_interior, y_interior)

    # Map interior points to FD solution indices
    x_indices = torch.round((x_interior - h) * (N-1) / (1 - 2*h)).long().clamp(0, N-1).cpu()
    y_indices = torch.round((y_interior - h) * (N-1) / (1 - 2*h)).long().clamp(0, N-1).cpu()
    
    # Extract corresponding FD solution values
    u_fd_values = torch.tensor(
        [u_fd[x_idx, y_idx] for x_idx, y_idx in zip(x_indices, y_indices)],
        dtype=torch.float32
    ).view(-1, 1).to(device)
    
    # Create boundary points
    n_boundary = 52
    # Bottom edge (y=0)
    x_bottom = torch.linspace(0, 1, n_boundary).view(-1, 1).to(device)
    y_bottom = torch.zeros_like(x_bottom).to(device)

    # Top edge (y=1)
    x_top = torch.linspace(0, 1, n_boundary).view(-1, 1).to(device)
    y_top = torch.ones_like(x_top).to(device)
    
    # Left edge (x=0)
    y_left = torch.linspace(0, 1, n_boundary).view(-1, 1).to(device)
    x_left = torch.zeros_like(y_left).to(device)
    
    # Right edge (x=1)
    y_right = torch.linspace(0, 1, n_boundary).view(-1, 1).to(device)
    x_right = torch.ones_like(y_right).to(device)
    
    # Combine all boundary points
    x_boundary = torch.cat([x_bottom, x_top, x_left, x_right], dim=0)
    y_boundary = torch.cat([y_bottom, y_top, y_left, y_right], dim=0)
    
    # Create datasets and dataloaders
    interior_dataset = TensorDataset(x_interior, y_interior, f_values, u_fd_values)
    interior_loader = DataLoader(interior_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    boundary_dataset = TensorDataset(x_boundary, y_boundary)
    boundary_loader = DataLoader(boundary_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize lists to store losses
    epoch_losses = []
    mse_losses = []
    pde_losses = []
    bc_losses = []
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_pde_loss = 0.0
        epoch_bc_loss = 0.0
        
        # Train on interior points (PDE loss + MSE loss)
        for x_batch, y_batch, f_batch, u_fd_batch in interior_loader:
            optimizer.zero_grad()
            
            # Compute Laplacian
            laplacian = model.compute_laplacian(x_batch, y_batch)
            
            # PDE loss: -Δu = f  ->  -laplacian - f = 0
            pde_loss = nn.MSELoss()(laplacian, f_batch)
            
            # Predict solution
            u_pred =  model(x_batch, y_batch)

            # MSE loss: (u_sol - u_pred)**2
            mse_loss = nn.MSELoss()(u_pred, u_fd_batch)

            epoch_mse_loss += mse_loss.item()
            epoch_pde_loss += pde_loss.item()
            loss = pde_loss + mse_loss 

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Train on boundary points (Dirichlet BC loss)
        for x_batch, y_batch in boundary_loader:
            optimizer.zero_grad()
            
            # BC loss: u = 0 on boundary
            u_pred = model(x_batch, y_batch)
            bc_loss = torch.mean(u_pred**2)
            
            epoch_bc_loss += bc_loss.item()
            loss = bc_loss 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Log every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_pde_loss = epoch_pde_loss / len(interior_loader) 
            avg_mse_loss = epoch_mse_loss / len(interior_loader) 
            avg_bc_loss = epoch_bc_loss / len(boundary_loader)
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}, PDE Loss: {avg_pde_loss:.6f}, MSE Loss: {avg_mse_loss:.6f}, BC Loss: {avg_bc_loss:.6f}")
            
            # Store for plotting
            epoch_losses.append(epoch_loss)
            pde_losses.append(avg_pde_loss)
            mse_losses.append(avg_mse_loss)
            bc_losses.append(avg_bc_loss)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    pathlib.Path(SAVE_PATH_DIR).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH_DIR / f"pinn_a{a}_b{b}.pth")
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(100, EPOCHS+1, 100), pde_losses, label='PDE Loss')
    plt.semilogy(range(100, EPOCHS+1, 100), mse_losses, label='MSE Loss')
    plt.semilogy(range(100, EPOCHS+1, 100), bc_losses, label='BC Loss')
    plt.semilogy(range(100, EPOCHS+1, 100), epoch_losses, label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title(f'PINN Training Losses (a={a}, b={b})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"fig/pinn_loss_a{a}_b{b}.png")
    
    return model

# ------------------------------
# Evaluate and visualize solution
def evaluate_pinn(model, a, b, compare_with_fd=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate grid for visualization
    n_viz = 52
    x = torch.linspace(0, 1, n_viz).to(device)
    y = torch.linspace(0, 1, n_viz).to(device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Reshape for model input
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    
    # Compute PINN prediction
    model.eval()
    with torch.no_grad():
        u_pinn = model(x_flat, y_flat).reshape(n_viz, n_viz).cpu().numpy()
    
    # Compute source term for reference
    def source_term(x, y):
        return x.cpu().numpy() * np.sin(a * np.pi * y.cpu().numpy()) + y.cpu().numpy() * np.sin(b * np.pi * x.cpu().numpy())
    
    f_values = source_term(x_flat, y_flat).reshape(n_viz, n_viz)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot source term f(x,y)
    im0 = axes[0].imshow(f_values, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[0].set_title(f'Source term f(x,y), a={a}, b={b}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot PINN solution
    im1 = axes[1].imshow(u_pinn, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[1].set_title('PINN Solution u(x,y)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f"fig/pinn_solution_a{a}_b{b}.png")
    
    if compare_with_fd:
        # Load finite difference solution if available
        try:
            data_dir = pathlib.Path(__file__).parent / "data"
            data = np.load(data_dir / "poisson_data.npz")
            
            # Find the correct index for the provided a and b values
            a_values = range(10)
            b_values = range(10)
            idx = None
            
            for i, a_val in enumerate(a_values):
                for j, b_val in enumerate(b_values):
                    if a_val == a and b_val == b:
                        idx = i * len(b_values) + j
                        break
                if idx is not None:
                    break
            
            if idx is not None:
                u_fd = data['u'][idx]
                
                # Resize for comparison (from 50x50 to n_viz x n_viz)
                from scipy.ndimage import zoom
                u_fd_resized = zoom(u_fd, n_viz/50)
                
                # Compute error
                error = np.abs(u_pinn - u_fd_resized)
                
                # Visualize comparison
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                im0 = axes[0].imshow(u_pinn, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
                axes[0].set_title('PINN Solution')
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('y')
                plt.colorbar(im0, ax=axes[0])
                
                im1 = axes[1].imshow(u_fd_resized, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
                axes[1].set_title('Finite Difference Solution')
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('y')
                plt.colorbar(im1, ax=axes[1])
                
                im2 = axes[2].imshow(error, origin='lower', extent=[0, 1, 0, 1], cmap='hot')
                axes[2].set_title('Absolute Error')
                axes[2].set_xlabel('x')
                axes[2].set_ylabel('y')
                plt.colorbar(im2, ax=axes[2])
                
                plt.tight_layout()
                plt.savefig(f"fig/pinn_vs_fd_a{a}_b{b}.png")
                
                # Compute error metrics
                l2_error = np.sqrt(np.mean((u_pinn - u_fd_resized)**2))
                max_error = np.max(np.abs(u_pinn - u_fd_resized))
                
                print(f"L2 Error: {l2_error:.6e}")
                print(f"Max Error: {max_error:.6e}")
                
                return u_pinn, u_fd_resized, l2_error, max_error
            else:
                print("Could not find matching a and b values in the finite difference data")
        except Exception as e:
            print(f"Error loading or comparing with finite difference solution: {e}")
    
    return u_pinn

# ------------------------------
if __name__ == "__main__":
    # Example: Train and evaluate PINN
    model_pinn = train_pinn(a=9, b=9)
    evaluate_pinn(model_pinn, a=9, b=9, compare_with_fd=True)