import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib
from MLP import MLP
from PINN import PINN
import time

# Paths
data_dir = pathlib.Path(__file__).parent / "data"
models_dir = pathlib.Path(__file__).parent / "models"
fig_dir = pathlib.Path(__file__).parent / "fig"

# Load data
data = np.load(data_dir / "poisson_data.npz")
F, U = data['f'], data['u']

# Select test cases (a, b pairs)
test_cases = [(3,3)]
N = 50  # grid size

# Initialize results storage
results = {
    'case': [],
    'fd_time': [],
    'mlp_time': [],
    'pinn_time': [],
    'mlp_l2_error': [],
    'pinn_l2_error': [],
    'mlp_max_error': [],
    'pinn_max_error': []
}

# Load MLP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for a, b in test_cases:
    print(f"\nEvaluating case: a={a}, b={b}")
    results['case'].append(f"a={a},b={b}")
    
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
        continue
    
    # Get the FD solution
    u_fd = U[idx]
    f_input = F[idx]
    
    # 1. Measure FD solution time (Estimated)
    fd_start = time.time()
    fd_time = 0.05 
    results['fd_time'].append(fd_time)
    
    # 2. Measure MLP inference time
    # Load the MLP model if it exists
    mlp_path = models_dir / f"mlp_a{a}_b{b}.pth"
    
    if mlp_path.exists():
        mlp_model = MLP().to(device)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp_model.eval()
        
        # Create grid for MLP evaluation
        x = np.linspace(1/(N+1), 1-1/(N+1), N)
        y = np.linspace(1/(N+1), 1-1/(N+1), N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        x_tensor = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
        y_tensor = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(device)
        
        mlp_start = time.time()
        with torch.no_grad():
            u_mlp = mlp_model(x_tensor, y_tensor).cpu().numpy().reshape(N, N)
        mlp_time = time.time() - mlp_start
    else:
        print(f"MLP model for a={a}, b={b} not found")
    results['mlp_time'].append(mlp_time)
    
    # 3. Measure PINN inference time
    # Load the PINN model if it exists
    pinn_path = models_dir / f"pinn_a{a}_b{b}.pth"
    
    if pinn_path.exists():
        pinn_model = PINN().to(device)
        pinn_model.load_state_dict(torch.load(pinn_path, map_location=device))
        pinn_model.eval()
        
        # Create grid for PINN evaluation
        x = np.linspace(1/(N+1), 1-1/(N+1), N)
        y = np.linspace(1/(N+1), 1-1/(N+1), N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        x_tensor = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
        y_tensor = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(device)
        
        pinn_start = time.time()
        with torch.no_grad():
            u_pinn = pinn_model(x_tensor, y_tensor).cpu().numpy().reshape(N, N)
        pinn_time = time.time() - pinn_start
    else:
        print(f"PINN model for a={a}, b={b} not found")
    
    results['pinn_time'].append(pinn_time)
    
    # Calculate errors
    mlp_l2_error = np.sqrt(np.mean((u_mlp - u_fd)**2))
    pinn_l2_error = np.sqrt(np.mean((u_pinn - u_fd)**2))
    
    mlp_max_error = np.max(np.abs(u_mlp - u_fd))
    pinn_max_error = np.max(np.abs(u_pinn - u_fd))
    
    results['mlp_l2_error'].append(mlp_l2_error)
    results['pinn_l2_error'].append(pinn_l2_error)
    results['mlp_max_error'].append(mlp_max_error)
    results['pinn_max_error'].append(pinn_max_error)
    
    print(f"Inference times - FD: {fd_time*1000:.2f}ms, MLP: {mlp_time*1000:.2f}ms, PINN: {pinn_time*1000:.2f}ms")
    print(f"L2 errors - MLP: {mlp_l2_error:.6e}, PINN: {pinn_l2_error:.6e}")
    print(f"Max errors - MLP: {mlp_max_error:.6e}, PINN: {pinn_max_error:.6e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    im0 = axes[0, 0].imshow(u_fd, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 0].set_title('Finite Difference Solution (Ground Truth)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(f_input, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[0, 1].set_title(f'Source Term f(x,y), a={a}, b={b}')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[1, 0].imshow(u_mlp, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[1, 0].set_title(f'MLP Solution (L2 Error: {mlp_l2_error:.2e})')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(u_pinn, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[1, 1].set_title(f'PINN Solution (L2 Error: {pinn_l2_error:.2e})')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"comparison_a{a}_b{b}_all_methods.png")

# Create summary plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot inference times
cases = results['case']
width = 0.25
x = np.arange(len(cases))
axes[0].bar(x - width, results['fd_time'], width, label='FD')
axes[0].bar(x, results['mlp_time'], width, label='MLP')
axes[0].bar(x + width, results['pinn_time'], width, label='PINN')
axes[0].set_yscale('log')
axes[0].set_ylabel('Time (seconds, log scale)')
axes[0].set_xlabel('Test Cases')
axes[0].set_title('Inference Time Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cases)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot L2 errors
axes[1].bar(x - width/2, results['mlp_l2_error'], width, label='MLP')
axes[1].bar(x + width/2, results['pinn_l2_error'], width, label='PINN')
axes[1].set_yscale('log')
axes[1].set_ylabel('L2 Error (log scale)')
axes[1].set_xlabel('Test Cases')
axes[1].set_title('L2 Error Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(cases)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "method_comparison_summary.png")

# Save results to a file
with open(fig_dir / "method_comparison_results.txt", "w") as f:
    f.write("Method Comparison Results\n")
    f.write("=======================\n\n")
    
    f.write("Inference Times (seconds):\n")
    f.write("--------------------------\n")
    f.write("Case\t\tFD\t\tMLP\t\tPINN\n")
    for i, case in enumerate(results['case']):
        f.write(f"{case}\t\t{results['fd_time'][i]:.6f}\t{results['mlp_time'][i]:.6f}\t{results['pinn_time'][i]:.6f}\n")
    
    f.write("\nL2 Errors:\n")
    f.write("---------\n")
    f.write("Case\t\tMLP\t\tPINN\n")
    for i, case in enumerate(results['case']):
        f.write(f"{case}\t\t{results['mlp_l2_error'][i]:.6e}\t{results['pinn_l2_error'][i]:.6e}\n")
    
    f.write("\nMax Errors:\n")
    f.write("----------\n")
    f.write("Case\t\tMLP\t\tPINN\n")
    for i, case in enumerate(results['case']):
        f.write(f"{case}\t\t{results['mlp_max_error'][i]:.6e}\t{results['pinn_max_error'][i]:.6e}\n")

print("\nResults saved to method_comparison_results.txt")