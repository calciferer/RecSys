import numpy as np

# mx = np.array([[1, 2, 3],
#                [4, 5, 6],
#                [7, 8, 9]])
# print(mx.T)
# print(mx)
# print(mx[1][0])
# print(np.unique([1,1,2,3,2]))
# mx = np.matrix.
# print(np.eye(3))
# a = np.array([1, 7, 5, 3, 9])
# print(np.argsort(-a))
# print(np.argwhere(a == 5)[0][0])

# import random
# l = [5, 4, 2, 8, 3]
# random.shuffle(l)
# print(l)

import pandas as pd

# mx = np.array(range(1, 10)).reshape((3, 3))
# print(mx)
# df = pd.DataFrame(mx, index=['A', 'B', 'C'], columns=['D', 'E', 'F'])
# print(df)
# print(df[df['D'] > 3].index)
# for x in df:
#     print(df[x])
# from numpy import linalg
#
# a = np.array([1, 0, 0])
# b = np.array([1, 1, 0])
# print(np.dot(a,b))
# print(linalg.norm(a))
# print(linalg.norm(b))
# import numpy as np
# from sklearn.metrics import pairwise_distances

# A = np.array([[1, 0, 0, 0],
#               [1, 1, 0, 1],
#               [0, 1, 0, 0]])
# d = A.T @ A
# print(d, '\n')
# norm = (A * A).sum(0, keepdims=True) ** .5
# print(norm, '\n')
# print((d / norm / norm.T))
# from sklearn.metrics import pairwise_distances
# print(1-pairwise_distances(A, metric="cosine"))

# d = {1: {1, 2, 3}, 2: {1, 2, 4}}
# s = set()
# for v in d.values():
#     s.update(v)

import matplotlib.pyplot as plt

# M = pd.DataFrame(np.array(range(9)).reshape((3, 3)), index=["A", "B", "C"], columns=["D", "E", "F"])
# print(M)
# print(M.sum()['D'])
# M.plot()

# x = np.arange(1, 10)
# y = np.arange(1, 10)
# z = x ** 2

# plt.plot(x, y, 'r:', label='y=x')
# plt.plot(x, z, label='y=x^2')
# plt.plot(x, x ** 3, label='cubic')

# plt.xlabel('x label')
# plt.ylabel('y label')
#
# plt.title("Simple Plot")
#
# plt.legend()
#
# plt.show()
# plt.plot(x, z, 'r:', label='y=x^2')
# plt.show()

df = pd.DataFrame(columns=['precision','recall'])
df.loc[0] = 'a','b'
print(df['recall'])
print(df)
