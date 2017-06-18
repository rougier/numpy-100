
# 100 numpy exercises with suggestion

This is a collection of exercises that have been collected in the numpy mailing
list, on stack overflow and in the numpy documentation. I've also created some
to reach the 100 limit. The goal of this collection is to offer a quick
reference for both old and new users but also to provide a set of exercices for
those who teach.

If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>

#### 1. Import the numpy package under the name `np` (★☆☆) (suggestion: import … as ..)



#### 2. Print the numpy version and the configuration (★☆☆) (suggestion: np.\_\_verison\_\_, np.show\_config)



#### 3. Create a null vector of size 10 (★☆☆) (suggestion: np.zeros)



#### 4.  How to find the memory size of any array (★☆☆) (suggestion: size, itemsize)



#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆) (suggestion: np.info)



#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆) (suggestion: array\[4\])



#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆) (suggestion: np.arange)



#### 8.  Reverse a vector (first element becomes last) (★☆☆) (suggestion: array\[::-1\])



#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) (suggestion: reshape)



#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆) (suggestion: np.nonzero)



#### 11. Create a 3x3 identity matrix (★☆☆) (suggestion: np.eye)



#### 12. Create a 3x3x3 array with random values (★☆☆) (suggestion: np.random.random)



#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) (suggestion: min, max)



#### 14. Create a random vector of size 30 and find the mean value (★☆☆) (suggestion: mean)



#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆) (suggestion: array\[1:-1, 1:-1\])



#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆) (suggestion: np.pad)



#### 17. What is the result of the following expression? (★☆☆) (suggestion: NaN = not a number, inf = infinity)


```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
0.3 == 3 * 0.1
```

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) (suggestion: np.diag)



#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) (suggestion: array\[::2\])



#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (suggestion: np.unravel\_index)



#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆) (suggestion: np.tile)



#### 22. Normalize a 5x5 random matrix (★☆☆) (suggestion: (x - min) / (max - min))



#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆) (suggestion: np.dtype)



#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) (suggestion: np.dot | @)



#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆) (suggestion: >, <=)



#### 26. What is the output of the following script? (★☆☆) (suggestion: np.sum)


```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)


```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```

#### 28. What are the result of the following expressions?


```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

#### 29. How to round away from zero a float array ? (★☆☆) (suggestion: np.uniform, np.copysign, np.ceil, np.abs)



#### 30. How to find common values between two arrays? (★☆☆) (suggestion: np.intersect1d)



#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆) (suggestion: np.seterr, np.errstate)



#### 32. Is the following expressions true? (★☆☆) (suggestion: imaginary number)


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆) (suggestion: np.datetime64, np.timedelta64)



#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆) (suggestion: np.arange(dtype=datetime64\['D'\]))



#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆) (suggestion: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=))



#### 36. Extract the integer part of a random array using 5 different methods (★★☆) (suggestion: %, np.floor, np.ceil, astype, np.trunc)



#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) (suggestion: np.arange)



#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆) (suggestion: np.fromiter)



#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) (suggestion: np.linespace)



#### 40. Create a random vector of size 10 and sort it (★★☆) (suggestion: sort)



#### 41. How to sum a small array faster than np.sum? (★★☆) (suggestion: np.add.reduce)



#### 42. Consider two random array A and B, check if they are equal (★★☆) (suggestion: np.allclose, np.array\_equal)



#### 43. Make an array immutable (read-only) (★★☆) (suggestion: flags.writeable)



#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆) (suggestion: np.sqrt, np.arctan2)



#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆) (suggestion: argmax)



#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆) (suggestion: np.meshgrid)



####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (suggestion: np.subtract.outer)



#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆) (suggestion: np.iinfo, np.finfo, eps)



#### 49. How to print all the values of an array? (★★☆) (suggestion: np.set\_printoptions)



#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆) (suggestion: argmin)



#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆) (suggestion: dtype)



#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆) (suggestion: np.atleast\_2d, T, np.sqrt)



#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place? (suggestion: astype(copy=False))



#### 54. How to read the following file? (★★☆) (suggestion: np.genfromtxt)


```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆) (suggestion: np.ndenumerate, np.ndindex)



#### 56. Generate a generic 2D Gaussian-like array (★★☆) (suggestion: np.meshgrid, np.exp)



#### 57. How to randomly place p elements in a 2D array? (★★☆) (suggestion: np.put, np.random.choice)



#### 58. Subtract the mean of each row of a matrix (★★☆) (suggestion: mean(axis=,keepdims=))



#### 59. How to sort an array by the nth column? (★★☆) (suggestion: argsort)



#### 60. How to tell if a given 2D array has null columns? (★★☆) (suggestion: any, ~)



#### 61. Find the nearest value from a given value in an array (★★☆) (suggestion: np.abs, argmin, flat)



#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆) (suggestion: np.nditer)



#### 63. Create an array class that has a name attribute (★★☆) (suggestion: class method)



#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★) (suggestion: np.bincount | np.add.at)



#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★) (suggestion: np.bincount)



#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★) (suggestion: np.unique)



#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★) (suggestion: sum(axis=(-2,-1)))



#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★) (suggestion: np.bincount)



#### 69. How to get the diagonal of a dot product? (★★★) (suggestion: np.diag)



#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★) (suggestion: array\[::4\])



#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★) (suggestion: array\[:, :, None\])



#### 72. How to swap two rows of an array? (★★★) (suggestion: array\[\[\]\] = array\[\[\]\])



#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★) (suggestion: repeat, np.roll, np.sort, view, np.unique)



#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★) (suggestion: np.repeat)



#### 75. How to compute averages using a sliding window over an array? (★★★) (suggestion: np.cumsum)



#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★) (suggestion: from numpy.lib import stride\_tricks)



#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★) (suggestion: np.logical_not, np.negative)



#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)



#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)



#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★) (suggestion: minimum, maximum)



#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★) (suggestion: stride\_tricks.as\_strided)



#### 82. Compute a matrix rank (★★★) (suggestion: np.linalg.svd)



#### 83. How to find the most frequent value in an array? (suggestion: np.bincount, argmax)



#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) (suggestion: stride\_tricks.as\_strided)



#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★) (suggestion: class method)



#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★) (suggestion: np.tensordot)



#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★) (suggestion: np.add.reduceat)



#### 88. How to implement the Game of Life using numpy arrays? (★★★)



#### 89. How to get the n largest values of an array (★★★) (suggestion: np.argsort | np.argpartition)



#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★) (suggestion: np.indices)



#### 91. How to create a record array from a regular array? (★★★) (suggestion: np.core.records.fromarrays)



#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★) (suggestion: np.power, \*, np.einsum)



#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★) (suggestion: np.where)



#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)



#### 95. Convert a vector of ints into a matrix binary representation (★★★) (suggestion: np.unpackbits)



#### 96. Given a two dimensional array, how to extract unique rows? (★★★) (suggestion: np.ascontiguousarray)



#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★) (suggestion: np.einsum)



#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)? (suggestion: np.cumsum, np.interp)



#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★) (suggestion: np.logical\_and.reduce, np.mod)



#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★) (suggestion: np.percentile)

