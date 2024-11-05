import numpy as np
import matplotlib.pyplot as plt

def fem_solver(N):
    dx = 1 / (N + 1)
    
    # get x
    x = np.array([j * dx if j % 2 == 0 else (j - 0.5) * dx for j in range(N + 2)])
    
    # initialize
    A = np.zeros((N, N))
    f = np.full(N, 2 * dx)  

    # stiffness matrix
    for i in range(N):
        if i == 0:
            A[i, i] = 8 / (3 * dx)
            A[i, i + 1] = -2 / (3 * dx)
        elif i == N - 1:
            A[i, i - 1] = -2 / dx
            A[i, i] = 8 / (3 * dx)
        else:
            if (i + 1) % 2 == 0:  # even
                A[i, i - 1] = -2 / (3 * dx)
                A[i, i] = 8 / (3 * dx)
                A[i, i + 1] = -2 / dx
            else:  # odd
                A[i, i - 1] = -2 / dx
                A[i, i] = 8 / (3 * dx)
                A[i, i + 1] = -2 / (3 * dx)

    u = np.linalg.solve(A, f)

    # B.C.
    u = np.concatenate(([0], u, [0]))

    # exact solution
    u_exact = x * (1 - x)

    # calculate error
    error = np.max(np.abs(u - u_exact))
    
    return x, u, u_exact, error

N_values = [39, 79, 159, 319]
errors = []

plt.figure(figsize=(10, 6))
for N in N_values:
    x, u_fem, u_exact, error = fem_solver(N)
    errors.append(error)
    plt.plot(x, u_fem, label=f'N = {N}')

# exact solution
plt.plot(x, u_exact, 'k--', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('FEM Solution for Different N Values')
plt.show()

# error plot
plt.figure(figsize=(8, 5))
plt.plot(N_values, errors, 'o-')
plt.xlabel('N')
plt.ylabel('Max Error')
plt.title('Error vs. N')
plt.show()
