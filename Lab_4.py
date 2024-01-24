import numpy as np


def lu(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (matrix[k][i] - sum) / U[i][i]

    return L, U


def solve_lu(L, U, b):
    n = len(L)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return x


matrix = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
                   [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
                   [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
                   [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
                   [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
                   [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
                   [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])
L, U = lu(matrix)
x = np.array(solve_lu(L, U, b))
print(x, '\n')

print(f'L: {L}', '\n')

print(f'U: {U}', '\n')

print(x, '\n')

result = np.array(matrix @ x)

print('Погрешность: ', max(abs(result - b)))
