import numpy as np


def find_pivot(A, p, n):
    max_row = p
    max_val = abs(A[p][p])

    for i in range(p + 1, n):
        if abs(A[i][p]) > max_val:
            max_row = i
            max_val = abs(A[i][p])

    return max_row


def gaussian_elimination(A, b):
    n = len(b)
    x = np.zeros(n, dtype=float)

    # Прямой ход метода Гаусса
    for p in range(n - 1):
        pivot_row = find_pivot(A, p, n)

        A[[p, pivot_row]] = A[[pivot_row, p]]
        b[[p, pivot_row]] = b[[pivot_row, p]]

        for i in range(p + 1, n):
            factor = A[i][p] / A[p][p]

            for j in range(p, n):
                A[i][j] = A[i][j] - factor * A[p][j]

            b[i] = b[i] - factor * b[p]

    # Обратный ход метода Гаусса
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        sum_val = np.sum(A[i][i + 1:n] * x[i + 1:n])
        x[i] = (b[i] - sum_val) / A[i][i]

    return x


A = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

x = gaussian_elimination(A, b)
print("Решение уравнения: ", x)

result = A @ x

print('Погрешность: ', max(abs(result - b)))
