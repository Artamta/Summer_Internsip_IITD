import numpy as np

#matrix dimensins:
matrix_dims=9

# Creating a matrix:
matrix = np.zeros((matrix_dims,matrix_dims))


#half_in_dims
half_path= matrix_dims/2

matrix[5,5]=1

print(matrix)

    