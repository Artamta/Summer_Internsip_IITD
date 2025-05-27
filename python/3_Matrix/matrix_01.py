import numpy as np
import matplotlib.pyplot as plt

matrix_dims = 100
center = matrix_dims // 2
radius = 20

matrix = np.zeros((matrix_dims, matrix_dims))

for i in range(matrix_dims):
    for j in range(matrix_dims):
        dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
        if dist <= radius:
            matrix[i, j] = 1

print(matrix)

# Plot the matrix as an image
plt.imshow(matrix, cmap='gray')
plt.title('Bullseye (Filled Circle) in 100x100 Matrix')
plt.axis('off')
plt.show()