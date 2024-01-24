import numpy as np
import cmath

A = np.array([[2.2, 4, -3, 1.5, 0.6, 2, 0.7],
              [4, 3.2, 1.5, -0.7, -0.8, 3, 1],
              [-3, 1.5, 1.8, 0.9, 3, 2, 2],
              [1.5, -0.7, 0.9, 2.2, 4, 3, 1],
              [0.6, -0.8, 3, 4, 3.2, 0.6, 0.7],
              [2, 3, 2, 3, 0.6, 2.2, 4],
              [0.7, 1, 2, 1, 0.7, 4, 3.2]])
B = np.array([3.2, 4.3, -0.1, 3.5, 5.3, 9.0, 3.7])

n = len(A)
Y = np.zeros(n, dtype=(object))
X = np.zeros(n, dtype=(object))
S = np.zeros((n, n), dtype=(object))

for i in range(n):
    for j in range(n):

        if (i == 0):
            if (j == 0):
                S[i][j] = cmath.sqrt(A[i][j])
            else:
                if (j > i):
                    S[i][j] = A[0][j] / S[0][0]
        else:
            if (i == j):
                S[i][i] = cmath.sqrt(A[i][i] - sum((S[k][i]) ** 2 for k in range(0, i)))
            else:
                if (j > i):
                    S[i][j] = (A[i][j] - sum(S[k][i] * S[k][j] for k in range(0, i))) / S[i][i]

Y[0] = B[0] / S[0][0]
for i in range(1, n):
    Y[i] = (B[i] - sum(S[k][i] * Y[k] for k in range(0, i))) / S[i][i]

X[n - 1] = Y[n - 1] / S[n - 1][n - 1]

for i in range(n - 2, -1, -1):  # 3,2,1,0
    X[i] = (Y[i] - sum(S[i][k] * X[k] for k in range(i + 1, n))) / S[i][i]

print(S)

print('Решение: ', X, '\n')

print('Погрешность: ', max(abs(A @ X - B)))
