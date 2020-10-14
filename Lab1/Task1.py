# import some packages we need
import numpy as np

# task 1

# Create a matrix
a = np.array([[2, 4, 5],[5, 2, 200]])

# Get the first row of the matrix
b = a[0 , :]

# Return 500 * 1 matrix of samples from the standard normal distribution
f = np.random.randn(500, 1)

# Get the values which are less than 0
g=f[f<0]

# Create a row of size 100 which is full of 0. And add 0.35 to each element of the row
x = np.zeros (100) + 0.35

# Create a row of size length of the x which is 100, and all elements of the column is 1
# Then, multiply every element of the column with 0.6
y = 0.6 * np.ones([1, len(x)])

# subtract x with y element-wise
z = x - y

# generate a numpy array
a = np.linspace(1,200)


b = a[::-1]

b[b <=50]=0