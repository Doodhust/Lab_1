import math
import numpy as np

np.set_printoptions(precision=7)

A = np.array([[2.12, 0.48, 1.34, 0.88, 11.172],
              [0.42, 3.95, 1.87, 0.43, 0.115],
              [1.34, 1.87, 2.98, 0.46, 9.009],
              [0.88, 0.43, 0.46, 4.44, 9.349]]).astype(float)
AllMatrix = [A]
print(A)
RMatrix = []
n = len(A)


def vectornormali(a, k):
    p_stepk_ = np.zeros(k)  # [0]
    p_stepk_k_ = np.zeros(n - k)  # [].. size = 2 [0,0]

    a = [row[k] for row in a]
    a = a[k:n]

    for i in range(n - k):  # 0..2  #1...2
        if (i == 0):
            if (a[i] >= 0):
                sigma_k = 1
                p_stepk_k_[0] = a[i] + sigma_k * (math.sqrt(sum(a[l] ** 2 for l in range(i, n - k))))
            else:
                sigma_k = -1
                p_stepk_k_[0] = a[i] + sigma_k * (math.sqrt(sum(a[l] ** 2 for l in range(i, n - k))))
        else:
            p_stepk_k_[i] = a[i]  # p_stepk_k_[1] = a[1][0]

    p_stepk = np.hstack((p_stepk_, p_stepk_k_))
    return p_stepk


def solve_r(a_, k, r, p_stepk):
    for i in range(k + 1, n + 1):  # 1,2,3
        a_for_dot = [row[i] for row in a_]
        for j in range(n):  # 0,1,2
            r[i - 1][j] = 2 * p_stepk.dot(a_for_dot) * p_stepk[j] / (sum(p_stepk[l] ** 2 for l in range(k, n)))
    r = np.transpose(r)

    return r


def solve_a(a_, p_stepk, k):
    new_a = np.zeros((n, n + 1))
    if (k > 0):
        for i in range(0, k):
            new_a[i] = a_[i]

    sigma = 1
    new_a[k][k] = -sigma * math.sqrt(sum(a_[l][k] ** 2 for l in range(k, n)))

    for i in range(k, n):
        for j in range(k + 1, n + 1):
            new_a[i][j] = a_[i][j] - 2 * p_stepk[i] * (
                    (sum(p_stepk[l] * a_[l][j] for l in range(k, n))) / (sum(p_stepk[l] ** 2 for l in range(k, n))))
    return new_a


def solve_x(g, r):
    x = np.zeros(n)

    x[n - 1] = g[n - 1] / r[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (g[i] - sum(r[i][j] * x[j] for j in range(i + 1, n))) / r[i][i]

    return x


for k in range(n - 1):
    a_ = np.copy(A)
    p_stepk = vectornormali(a_, k)
    s = np.linalg.norm((p_stepk)) ** 2

    r = np.zeros((n, n))
    r = solve_r(a_, k, r, p_stepk)
    RMatrix.append(r)

    A = solve_a(a_, p_stepk, k)
    AllMatrix.append(A)

A = AllMatrix[n - 1]
R = A[:, :n]
b = [row[-1] for row in A]
x = solve_x(b, R)
print("Решение: ", x)

A = AllMatrix[0]
a = A[:, :n]
b = [row[-1] for row in A]
print(np.linalg.solve(a, b))

solution = np.dot(a, x)

print("Погрешность: ", max(abs(solution - b)))
