


# 100 numpy exercises

This is a collection of exercises that have been collected in the numpy mailing list, on stack overflow
and in the numpy documentation. The goal of this collection is to offer a quick reference for both old
and new users but also to provide a set of exercises for those who teach.


If you find an error or think you've a better way to solve some of them, feel
free to open an issue at <https://github.com/rougier/numpy-100>.
File automatically generated. See the documentation to update questions/answers/hints programmatically.

#### 1. Import the numpy package under the name `np` (★☆☆)
`hint: import … as`
#### 2. Print the numpy version and the configuration (★☆☆)
`hint: np.__version__, np.show_config)`
#### 3. Create a null vector of size 10 (★☆☆)
`hint: np.zeros`
#### 4. How to find the memory size of any array (★☆☆)
`hint: size, itemsize`
#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)
`hint: np.info`
#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
`hint: array[4]`
#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)
`hint: arange`
#### 8. Reverse a vector (first element becomes last) (★☆☆)
`hint: array[::-1]`
#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
`hint: reshape`
#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)
`hint: np.nonzero`
#### 11. Create a 3x3 identity matrix (★☆☆)
`hint: np.eye`
#### 12. Create a 3x3x3 array with random values (★☆☆)
`hint: np.random.random`
#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
`hint: min, max`
#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
`hint: mean`
#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
`hint: array[1:-1, 1:-1]`
#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
`hint: np.pad`
#### 17. What is the result of the following expression? (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```
`hint: NaN = not a number, inf = infinity`
#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
`hint: np.diag`
#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
`hint: array[::2]`
#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
`hint: np.unravel_index`
#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
`hint: np.tile`
#### 22. Normalize a 5x5 random matrix (★☆☆)
`hint: (x -mean)/std`
#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
`hint: np.dtype`
#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
`hint:`
#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
`hint: >, <`
#### 26. What is the output of the following script? (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
`hint: np.sum`
#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```
`No hints provided...`
#### 28. What are the result of the following expressions? (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
`No hints provided...`
#### 29. How to round away from zero a float array ? (★☆☆)
`hint: np.uniform, np.copysign, np.ceil, np.abs, np.where`
#### 30. How to find common values between two arrays? (★☆☆)
`hint: np.intersect1d`
#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
`hint: np.seterr, np.errstate`
#### 32. Is the following expressions true? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
`hint: imaginary number`
#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
`hint: np.datetime64, np.timedelta64`
#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
`hint: np.arange(dtype=datetime64['D'])`
#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
`hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)`
#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
`hint: %, np.floor, astype, np.trunc`
#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
`hint: np.arange`
#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
`hint: np.fromiter`
#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
`hint: np.linspace`
#### 40. Create a random vector of size 10 and sort it (★★☆)
`hint: sort`
#### 41. How to sum a small array faster than np.sum? (★★☆)
`hint: np.add.reduce`
#### 42. Consider two random array A and B, check if they are equal (★★☆)
`hint: np.allclose, np.array_equal`
#### 43. Make an array immutable (read-only) (★★☆)
`hint: flags.writeable`
#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
`hint: np.sqrt, np.arctan2`
#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
`hint: argmax`
#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)
`hint: np.meshgrid`
#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)
`hint: np.subtract.outer`
#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)
`hint: np.iinfo, np.finfo, eps`
#### 49. How to print all the values of an array? (★★☆)
`hint: np.set_printoptions`
#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
`hint: argmin`
#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
`hint: dtype`
#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
`hint: np.atleast_2d, T, np.sqrt`
#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?
`hint: view and [:] =`
#### 54. How to read the following file? (★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
`hint: np.genfromtxt`
#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
`hint: np.ndenumerate, np.ndindex`
#### 56. Generate a generic 2D Gaussian-like array (★★☆)
`hint: np.meshgrid, np.exp`
#### 57. How to randomly place p elements in a 2D array? (★★☆)
`hint: np.put, np.random.choice`
#### 58. Subtract the mean of each row of a matrix (★★☆)
`hint: mean(axis=,keepdims=)`
#### 59. How to sort an array by the nth column? (★★☆)
`hint: argsort`
#### 60. How to tell if a given 2D array has null columns? (★★☆)
`hint: any, ~`
#### 61. Find the nearest value from a given value in an array (★★☆)
`hint: np.abs, argmin, flat`
#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
`hint: np.nditer`
#### 63. Create an array class that has a name attribute (★★☆)
`hint: class method`
#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
`hint: np.bincount | np.add.at`
#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
`hint: np.bincount`
#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
`hint: np.unique`
#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
`hint: sum(axis=(-2,-1))`
#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)
`hint: np.bincount`
#### 69. How to get the diagonal of a dot product? (★★★)
`hint: np.diag`
#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
`hint: array[::4]`
#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)
`hint: array[:, :, None]`
#### 72. How to swap two rows of an array? (★★★)
`hint: array[[]] = array[[]]`
#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)
`hint: repeat, np.roll, np.sort, view, np.unique`
#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
`hint: np.repeat`
#### 75. How to compute averages using a sliding window over an array? (★★★)
`hint: np.cumsum, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
`hint: from numpy.lib import stride_tricks, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
`hint: np.logical_not, np.negative`
#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
`No hints provided...`
#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
`No hints provided...`
#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)
`hint: minimum maximum`
#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 82. Compute a matrix rank (★★★)
`hint: np.linalg.svd, np.linalg.matrix_rank`
#### 83. How to find the most frequent value in an array?
`hint: np.bincount, argmax`
#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
`hint: stride_tricks.as_strided, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
`hint: class method`
#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
`hint: np.tensordot`
#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
`hint: np.add.reduceat, from numpy.lib.stride_tricks import sliding_window_view (np>=1.20.0)`
#### 88. How to implement the Game of Life using numpy arrays? (★★★)
`No hints provided...`
#### 89. How to get the n largest values of an array (★★★)
`hint: np.argsort | np.argpartition`
#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)
`hint: np.indices`
#### 91. How to create a record array from a regular array? (★★★)
`hint: np.core.records.fromarrays`
#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)
`hint: np.power, *, np.einsum`
#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)
`hint: np.where`
#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)
`No hints provided...`
#### 95. Convert a vector of ints into a matrix binary representation (★★★)
`hint: np.unpackbits`
#### 96. Given a two dimensional array, how to extract unique rows? (★★★)
`hint: np.ascontiguousarray | np.unique`
#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)
`hint: np.einsum`
#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?
`hint: np.cumsum, np.interp`
#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)
`hint: np.logical_and.reduce, np.mod`
#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)
`hint: np.percentile`