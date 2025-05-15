import numpy as np

matrix_dims = 11
center = matrix_dims // 2
radius = 2

matrix = np.zeros((matrix_dims, matrix_dims))

for i in range(matrix_dims):
    for j in range(matrix_dims):
        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
        if dist <= radius:
            matrix[i, j] = 1

print(matrix)