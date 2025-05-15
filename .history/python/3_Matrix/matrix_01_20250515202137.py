import numpy as np

#matrix dimensins:
matrix_dims=10

# Creating a matrix:
matrix = np.zeros((matrix_dims,matrix_dims))
b= np.zeros([5,5])
b[1,1]=1
print(b)

#half_in_dims
half_path= matrix_dims/2

matrix[half_path,half_path]=1

print(matrix)

    