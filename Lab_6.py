import numpy as np

np.set_printoptions(precision=7, suppress=True)


def get_inv(A, depth=0):
    n = len(A)
    k = n - 1

    if n == 1:
        return np.matrix([[1.0 / A[0, 0]]])

    Ap = A[:k, :k]
    V, U = A[k, :k], A[:k, k].reshape(-1, 1)

    Ap_inv = get_inv(Ap, depth + 1)

    alpha = 1.0 / (A[k, k] - np.dot(V, np.dot(Ap_inv, U))).item()
    Q = -np.dot(V, Ap_inv) * alpha
    P = Ap_inv - np.dot(np.dot(Ap_inv, U), Q)
    R = -np.dot(Ap_inv, U) * alpha

    A_inv = np.zeros((n, n))
    A_inv[:k, :k] = P
    A_inv[k, :k] = Q
    A_inv[:k, k] = R.T
    A_inv[k, k] = alpha

    return A_inv


a1 = np.matrix([[3, 1, 0],
                [3, 2, 1],
                [1, 1, 1]]).astype(float)
b = np.matrix([2, 1, 1]).astype(float)
n = len(a1)
a_1 = get_inv(a1)

print("Обратная матрица методом окаймления: ", a_1)

x = a_1 * b.T

print("Решение: ", x, '\n')

solution = a1 * x

print('Погрешность: ', max(abs(np.ravel(solution) - np.ravel(b))))
