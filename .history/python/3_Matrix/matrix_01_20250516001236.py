import numpy as np

matrix_dims = 10
center = matrix_dims // 2
radius = 2 

matrix = np.zeros((matrix_dims, matrix_dims))

for i in range(matrix_dims):
    for j in range(matrix_dims):
        # Calculate distance from center
        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
        # Set value to 1 if within the radius
        if dist <= radius:
            matrix[i, j] = 1

print(matrix)