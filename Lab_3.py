import numpy as np

A = np.array([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])

b = np.array([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])


def optimal_elimination(matrix, b):
    n = len(matrix)
    A = np.hstack((matrix, b.reshape(-1, 1)))
    A[0, :] /= A[0, 0]

    for k in range(n - 1):
        A_temp = A.copy()
        for p in range(k + 2, n + 1):
            s1 = np.dot(A_temp[:k + 1, p], A_temp[k + 1, :k + 1])
            s2 = np.dot(A_temp[:k + 1, k + 1], A_temp[k + 1, :k + 1])
            A[k + 1, p] = (A_temp[k + 1, p] - s1) / (A_temp[k + 1, k + 1] - s2)
            for i in range(k + 1):
                A[i, p] = A_temp[i, p] - A[k + 1, p] * A_temp[i, k + 1]
        for j in range(k + 1):
            A[k + 1, j] = 0
            A[j, k + 1] = 0
        A[k + 1, k + 1] = 1
    print(A)
    return A[:, n]


x = optimal_elimination(A, b)

result = A @ x

print('Решение: ', x, '\n')

print('Погрешность: ', max(abs(result - b)))
