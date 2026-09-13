import numpy as np

M = 100000

c = np.array([-3, -2, 0, 0, -M, -M], dtype=float)

A = np.array([
    [500, 300, -1, 0, 1, 0],
    [20, 10, 0, -1, 0, 1]
], dtype=float)

b = np.array([2000, 80], dtype=float)

table = np.c_[A, b]
basis = [4, 5]

z = np.r_[-c, 0.]

for i in range(2):
    z += c[basis[i]] * np.r_[table[i]]

while True:
    col = np.argmin(z[:-1])

    if z[col] >= 0:
        break

    ratios = []

    for i in range(len(table)):
        if table[i, col] > 0:
            ratios.append(table[i, -1] / table[i, col])
        else:
            ratios.append(np.inf)

    row = np.argmin(ratios)

    pivot = table[row, col]
    table[row] /= pivot

    for i in range(len(table)):
        if i != row:
            table[i] -= table[i, col] * table[row]

    z -= z[col] * table[row]
    basis[row] = col

x = np.zeros(6)

for i in range(len(basis)):
    x[basis[i]] = table[i, -1]

print("Animal Feed Optimization")
print("Corn =", round(x[0], 2), "kg")
print("Soybean =", round(x[1], 2), "kg")
print("Minimum Cost = ₹", round(-z[-1], 2))
