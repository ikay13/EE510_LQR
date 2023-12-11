import numpy as np
#from slycot import control


def find_coefficients(eigenvalues):
    # Create a polynomial with the given roots (eigenvalues)
    polynomial = np.poly(eigenvalues)
    # The coefficients are returned in descending order
    return polynomial[1:]



def characteristic_polynomial_manual(A):
    # Assuming A is a 3x3 matrix
    # The coefficients of the characteristic polynomial (excluding the highest term) can be calculated as follows:
    # c2 = - (trace of A)
    # c1 = sum of 2x2 principal minors of A
    # c0 = - (determinant of A)

    # Calculate c2
    c2 = - (A[0][0] + A[1][1] + A[2][2])

    # Calculate c1
    minor1 = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    minor2 = A[0][0] * A[2][2] - A[0][2] * A[2][0]
    minor3 = A[1][1] * A[2][2] - A[1][2] * A[2][1]
    c1 = minor1 + minor2 + minor3

    # Calculate c0
    c0 = - (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
            A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
            A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

    return [c2, c1, c0]

def compute_controllability_matrix(A, B):
    n = A.shape[0]
    Gc = B
    for i in range(1, n):
        Gc = np.hstack((Gc, np.linalg.matrix_power(A, i) @ B))
    return Gc

def compute_transformation_matrix(A, B, alpha):
    Gc = compute_controllability_matrix(A, B)
    P = np.linalg.inv(Gc)
    Q = np.zeros_like(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[i, j] = 1
            elif j > i:
                Q[i, j] = alpha[j - i - 1]
    return Q

def compute_A_bar_BK(A, B, P, Q, alpha_bar, alpha):
    # Compute A_bar and B_bar
    A_bar = P @ A @ np.linalg.inv(P)
    B_bar = P @ B

    # Compute K_bar
    K_bar = np.array([a_bar - a for a_bar, a in zip(alpha_bar, alpha)])

    # Ensure K_bar is reshaped to match the dimensions of B_bar for matrix multiplication
    K_bar_reshaped = K_bar.reshape(1, -1)

    # Compute (A_bar - B_bar * K)
    A_bar_BK = A_bar - B_bar @ K_bar_reshaped

    return A_bar_BK

def compute_K(K_bar, P):
    # Compute K = K_bar * P
    K = K_bar @ P
    return K

def compute_A_k(A, B, K):
    # Compute A_k = A - B * K
    K = K.reshape(1, -1)
    A_k = A - B @ K
    return A_k

# Example usage with eigenvalues R1 = [−1, −2 + 2j, −2 − 2j]
eigenvalues_R1 = [-1, -2 + 2j, -2 - 2j]
coefficients = find_coefficients(eigenvalues_R1)
print("Coefficients of Δ_d(s):", coefficients)


# Example usage with matrix A1
A1 = [[0, 1, 0], [1, -1, 1], [0, 1, 0]]
char_poly_A1 = characteristic_polynomial_manual(A1)
print("Characteristic polynomial of A1 (excluding highest term):", char_poly_A1)

# Placeholder for Gc and coefficients
Gc_placeholder = np.identity(3)  # Replace with actual Gc
coefficients_placeholder = find_coefficients(eigenvalues_R1)  # Replace with actual coefficients from previous steps


# Placeholder for matrices A, B, and coefficients
A_placeholder = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])  # Replace with actual A
B_placeholder = np.array([[1], [1], [0]])  # Replace with actual B
alpha_bar_placeholder = [4, 14, 8]  # Replace with actual alpha_bar from previous steps
alpha_placeholder = [1, 2, 3]  # Replace with actual alpha from previous steps

# Compute the transformation matrix Q
Q = compute_transformation_matrix(A_placeholder, B_placeholder, coefficients_placeholder)
print("Transformation Matrix Q:\n", Q)

# ... rest of your code ...

A_bar_BK = compute_A_bar_BK(A_placeholder, B_placeholder, np.linalg.inv(Q), Q, alpha_bar_placeholder, alpha_placeholder)
print("Matrix (A_bar - B_bar * K):\n", A_bar_BK)

# Assuming K_bar and P are from previous steps
K_bar1 = np.array([4, 14, 8])  # K_bar1
P_placeholder = np.linalg.inv(Q)  # Inverse of Q from Step 3

K1 = compute_K(K_bar1, P_placeholder)
print("Matrix K1:\n", K1)

A_k1 = compute_A_k(A_placeholder, B_placeholder, K1)
print("Matrix A_k1:\n", A_k1)

# Compute eigenvalues of A_k1
eigenvalues_A_k1 = np.linalg.eigvals(A_k1)
print("Eigen value group 1:", eigenvalues_A_k1)
