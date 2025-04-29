import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import pathlib

data_dir = pathlib.Path(__name__).parent / "data"

# Paramètres
N = 50  # Nombre de points intérieurs dans chaque direction (50x50 = 2500 points)
a_values = [i  for i in range(10)] 
b_values = [i for i in range(10)]   

# Discrétisation
h = 1.0 / (N + 1)
x = np.linspace(h, 1 - h, N)
y = np.linspace(h, 1 - h, N)
X, Y = np.meshgrid(x, y, indexing='ij')

F,U = [],[]

for a in a_values:
    for b in b_values:
        f = X * np.sin(a * np.pi * Y) + Y * np.sin(b * np.pi * X)
        f = f.reshape(N * N)

        # Construction de la matrice A
        main_diag = -2.0 * np.ones(N)
        off_diag = np.ones(N - 1)
        T = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N))
        I = identity(N) 
        A = (kron(I, T) + kron(T, I)) / (h ** 2)

        # Résolution du système
        u = spsolve(A, f)

        # Reshape
        u = u.reshape((N, N))
        f = f.reshape((N, N))

        F.append(f)
        U.append(u)

np.savez(data_dir / "poisson_data.npz" , f=F, u=U)
