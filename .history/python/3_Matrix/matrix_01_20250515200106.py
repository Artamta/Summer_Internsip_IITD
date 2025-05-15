import numpy as np

#matrix dimensins:
matrix_dims=10

# Creating a matrix:

matrix = np.zeros((matrix_dims,matrix_dims))
print(matrix)

half_path= matrix_dims/2

for i in range(half_path):
    
    