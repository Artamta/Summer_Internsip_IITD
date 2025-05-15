import numpy as np

#matrix dimensins:
matrix_dims=10

# Creating a matrix:
matrix = np.zeros((matrix_dims,matrix_dims))
print(matrix)

#half_in_dims
half_path= matrix_dims/2

matrix(half_path+1,half_path)=1
matrix(half_path,half_path+1)=1
matrix(half_path-1,half_path)=1
matrix(half_path,half_path-1)=1
    
    
print(matrix)