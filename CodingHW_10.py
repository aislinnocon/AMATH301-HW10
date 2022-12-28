import numpy as np


####### Problem 1 ######
A = np.array([[1,-2,0,-1], [4,0,-2,1]])
# print(A)

U, s, Vt = np.linalg.svd(A, full_matrices=False)
V = Vt.T
A1 = V.copy()
# print("A1 = ", A1)

U, s, Vt = np.linalg.svd(A)
V = Vt.T
A2 = V
# print("A2 = ", A2)

A3 = 2
# print("A3 = ", A3)

k = 1
Uk = U[:, :k]
Vk = V[:, :k]
Sk = np.diag(s[:k])
reconstructed_data = Uk @ Sk @ Vk.T
A4 = reconstructed_data.copy()
# print("A4 = ", A4)

E = np.cumsum(s ** 2) / np.sum(s ** 2)
A5 = E[0]
# print("A5 = ", A5)

###### Problem 2 ########
n = 114
q = -np.ones(n-1)
Q = np.diag(q, -1)
Z = np.diag(q, 1)
u = 2*np.ones(n)
U = np.diag(u)
An = U + Q + Z
P = np.zeros((n, 1))
for k in range(1, n + 1):
    P[k - 1] = 2 * (1 - np.cos((53 * np.pi)/115)) * np.sin((53 * np.pi * k)/115)
U, s, Vt = np.linalg.svd(An)
V = Vt.T

A6 = s.reshape(1, 114)
# print("A6 = ", A6)

A7 = U.T
# print("A7 = ", A7)

A8 = Vt.T
# print("A8 = ", A8)

sI = np.diag(1 / s)
A9 = sI
# print("A9 = ", A9)

A10 = U.T @ P
#print("A10 = ", A10)

A11 = A9 @ A10
#print("A11 = ", A11)

A12 = Vt.T @ A11
#print("A12 = ", A12)

######## Problem 3 #########
data = np.genfromtxt('hw10_img.csv', delimiter=',')
m, n = data.shape
U, s, Vt = np.linalg.svd(data, full_matrices=False)
V = Vt.T

A13 = m * n * 8 / 1e6
print("A13 = ", A13)

E = np.cumsum(s ** 2) / np.sum(s ** 2)
for k in range(E.size):
    if (E[k] > 0.99):
        break
A14 = k + 1
# print("A14 = ", A14)

A15 = A14 * (m + n + 1) * 8 / 1000000
print("A15 = ", A15)

