import numpy as np
import time

start_time = time.time()

A = np.matrix([[0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
              [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
              [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
              [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
              [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
              [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
              [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105]])

b = np.matrix([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673])

def QR(A):
    n = A.shape[0]
    R = A.copy()
    Q = np.matrix(np.eye(n))
    for k in range(n):
        p = np.zeros(n)
        s = sum(a[0, 0] ** 2 for a in R[k:, k]) ** 0.5
        sigma = -1 + 2 * int(R[k, k] > 0) # -1 или +1
        p[k] = R[k, k] + sigma * s
        p[k + 1:] = R[k + 1:, k].T
        s = sum(a ** 2 for a in p)
        Pk = np.matrix(np.eye(n))
        for i in range(n):
            for j in range(n):
                Pk[i, j] -= 2 * p[i] * p[j] / s

        R = Pk * R
        Q = Q * Pk
    return Q, R


def solve_mirror(A, b, n):
    A = np.matrix(A)
    b = np.matrix(b)
    b = b.T
    Q, R = QR(A)
    n, *_ = A.shape
    g = np.squeeze(np.asarray(Q.T * b))
    x = np.zeros(n)
    x[n - 1] = g[n - 1] / R[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (g[i] - sum([R[i, j] * x[j] for j in range(i + 1, n)])) / R[i, i]

    return list(x)


x = solve_mirror(A, b, A.shape[0])

solution = np.dot(A, x)
print(A)

print('Решение: ', x)

print("Погрешность: ", max(abs(np.ravel(solution) - np.ravel(b))))

end_time = time.time()
execution_time = (end_time - start_time) * 1000.
print(f"Время выполнения программы: {execution_time} миллисекунд")

cond_number = np.linalg.cond(A)
print("Число обусловленности матрицы A:", cond_number)

