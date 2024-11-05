import numpy as np
import matplotlib.pyplot as plt

def fem_solver(N):
    # Step size
    dx = 1 / (N + 1)
    
    # Define mesh points according to the problem's instructions
    x = np.array([j * dx if j % 2 == 0 else (j - 0.5) * dx for j in range(N + 2)])
    
    # Initialize stiffness matrix A and load vector f
    A = np.zeros((N, N))
    f = np.full(N, 2 * dx)  # Right-hand side vector corresponding to -u_xx = 2

    # Assemble stiffness matrix A based on non-uniform mesh
    for i in range(N):
        if i == 0:
            A[i, i] = 8 / (3 * dx)
            A[i, i + 1] = -2 / (3 * dx)
        elif i == N - 1:
            A[i, i - 1] = -2 / dx
            A[i, i] = 8 / (3 * dx)
        else:
            if (i + 1) % 2 == 0:  # Even j
                A[i, i - 1] = -2 / (3 * dx)
                A[i, i] = 8 / (3 * dx)
                A[i, i + 1] = -2 / dx
            else:  # Odd j
                A[i, i - 1] = -2 / dx
                A[i, i] = 8 / (3 * dx)
                A[i, i + 1] = -2 / (3 * dx)

    # Solve the linear system
    u = np.linalg.solve(A, f)

    # Apply boundary conditions: u(0) = 0, u(1) = 0
    u = np.concatenate(([0], u, [0]))

    # Exact solution
    u_exact = x * (1 - x)

    # Calculate maximum error
    error = np.max(np.abs(u - u_exact))
    
    return x, u, u_exact, error

# FEM solution and error for different N values
N_values = [39, 79, 159, 319]
errors = []

plt.figure(figsize=(10, 6))
for N in N_values:
    x, u_fem, u_exact, error = fem_solver(N)
    errors.append(error)
    plt.plot(x, u_fem, label=f'N = {N}')

# Plot exact solution
plt.plot(x, u_exact, 'k--', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('FEM Solution for Different N Values')
plt.show()

# Error plot
plt.figure(figsize=(8, 5))
plt.plot(N_values, errors, 'o-')
plt.xlabel('N')
plt.ylabel('Max Error')
plt.title('Error vs. N')
plt.show()
