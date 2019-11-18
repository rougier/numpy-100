
qha = {
    "q1": "Import the numpy package under the name `np` (★☆☆)",
    "h1": "hint: import … as ",
    "a1":
"""
import numpy as np
""",
    "q2": "Print the numpy version and the configuration (★☆☆)",
    "h2": "hint: np.__version__, np.show_config)",
    "a2":
"""

""",
    "q3": "Create a null vector of size 10 (★☆☆)",
    "h3": "hint: np.zeros",
    "a3":
"""

""",
    "q4": "How to find the memory size of any array (★☆☆)",
    "h4": "hint: size, itemsize",
    "a4":
"""

""",
    "q5": "How to get the documentation of the numpy add function from the command line? (★☆☆)",
    "h5": "hint: np.info",
    "a5":
"""

""",
    "q6": "Create a null vector of size 10 but the fifth value which is 1 (★☆☆)",
    "h6": "hint: array[4]",
    "a6":
"""

""",
    "q7": "Create a vector with values ranging from 10 to 49 (★☆☆)",
    "h7": "hint: arange",
    "a7":
"""

""",
    "q8": "Reverse a vector (first element becomes last) (★☆☆)",
    "h8": "hint: array[::-1]",
    "a8":
"""

""",
    "q9": "Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)",
    "h9": "hint: reshape",
    "a9":
"""

""",
    "q10": "Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)",
    "h10": "hint: np.nonzero",
    "a10":
"""

""",
    "q11": "Create a 3x3 identity matrix (★☆☆)",
    "h11": "hint: np.eye",
    "a11":
"""

""",
    "q12": "Create a 3x3x3 array with random values (★☆☆)",
    "h12": "hint: np.random.random",
    "a12":
"""

""",
    "q13": "Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)",
    "h13": "hint: min, max",
    "a13":
"""

""",
    "q14": "Create a random vector of size 30 and find the mean value (★☆☆)",
    "h14": "hint: mean",
    "a14":
"""

""",
    "q15": "Create a 2d array with 1 on the border and 0 inside (★☆☆)",
    "h15": "hint: array[1:-1, 1:-1]",
    "a15":
"""

""",
    "q16": "How to add a border (filled with 0's) around an existing array? (★☆☆)",
    "h16": "hint: np.pad",
    "a16":
"""

""",
    "q17": """
What is the result of the following expression? (★☆☆)"
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
# 0.3 == 3 * 0.1
# ```""",
    "h17": "hint: NaN = not a number, inf = infinity",
    "a17":
"""

""",
    "q18": "Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)",
    "h18": "hint: np.diag",
    "a18":
"""

""",
    "q19": "Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)",
    "h19": "hint: array[::2]",
    "a19":
"""

""",
    "q20": "Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?",
    "h20": "hint: np.unravel_index",
    "a20":
"""

""",
    "q21": "Create a checkerboard 8x8 matrix using the tile function (★☆☆)",
    "h21": "hint: np.tile",
    "a21":
"""

""",
    "q22": "Normalize a 5x5 random matrix (★☆☆)",
    "h22": "hint: (x -mean)/std",
    "a22":
"""

""",
    "q23": "Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)",
    "h23": "hint: np.dtype",
    "a23":
"""

""",
    "q24": "Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)",
    "h24": "hint: ",
    "a24":
"""

""",
    "q25": "Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)",
    "h25": "hint: >, <=",
    "a25":
"""

""",
    "q26": """
What is the output of the following script? (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```
""",
    "h26": "hint: np.sum",
    "a26":
"""

""",
    "q27": """
Consider an integer vector Z, which of these expressions are legal? (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```""",
    "h27": "No hints provided...",
    "a27":
"""

""",
    "q28": """
What are the result of the following expressions?
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```
""",

    "h28": "No hints provided... ",
    "a28":
"""

""",
    "q29": "How to round away from zero a float array ? (★☆☆)",
    "h29": "hint: np.uniform, np.copysign, np.ceil, np.abs",
    "a29":
"""

""",
    "q30": "How to find common values between two arrays? (★☆☆)",
    "h30": "hint: np.intersect1d",
    "a30":
"""

""",
    "q31": "How to ignore all numpy warnings (not recommended)? (★☆☆)",
    "h31": "hint: np.seterr, np.errstate",
    "a31":
"""

""",
    "q32": """
Is the following expressions true? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```
""",
    "h32": "hint: imaginary number",
    "a32":
"""

""",
    "q33": "How to get the dates of yesterday, today and tomorrow? (★☆☆)",
    "h33": "hint: np.datetime64, np.timedelta64",
    "a33":
"""

""",
    "q34": "How to get all the dates corresponding to the month of July 2016? (★★☆)",
    "h34": "hint: np.arange(dtype=datetime64['D'])",
    "a34":
"""

""",
    "q35": "How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)",
    "h35": "hint: np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)",
    "a35":
"""

""",
    "q36": "Extract the integer part of a random array using 5 different methods (★★☆)",
    "h36": "hint: %, np.floor, np.ceil, astype, np.trunc",
    "a36":
"""

""",
    "q37": "Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)",
    "h37": "hint: np.arange",
    "a37":
"""

""",
    "q38": "Consider a generator function that generates 10 integers and use it to build an array (★☆☆)",
    "h38": "hint: np.fromiter",
    "a38":
"""

""",
    "q39": "Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)",
    "h39": "hint: np.linspace",
    "a39":
"""

""",
    "q40": "Create a random vector of size 10 and sort it (★★☆)",
    "h40": "hint: sort",
    "a40":
"""

""",
    "q41": "How to sum a small array faster than np.sum? (★★☆)",
    "h41": "hint: np.add.reduce",
    "a41":
"""

""",
    "q42": "Consider two random array A and B, check if they are equal (★★☆)",
    "h42": "hint: np.allclose, np.array_equal",
    "a42":
"""

""",
    "q43": "Make an array immutable (read-only) (★★☆)",
    "h43": "hint: flags.writeable",
    "a43":
"""

""",
    "q44": "Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)",
    "h44": "hint: np.sqrt, np.arctan2",
    "a44":
"""

""",
    "q45": "Create random vector of size 10 and replace the maximum value by 0 (★★☆)",
    "h45": "hint: argmax",
    "a45":
"""

""",
    "q46": "Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)",
    "h46": "hint: np.meshgrid",
    "a46":
"""

""",
    "q47": "Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))",
    "h47": "hint: np.subtract.outer",
    "a47":
"""

""",
    "q48": "Print the minimum and maximum representable value for each numpy scalar type (★★☆)",
    "h48": "hint: np.iinfo, np.finfo, eps",
    "a48":
"""

""",
    "q49": "How to print all the values of an array? (★★☆)",
    "h49": "hint: np.set_printoptions",
    "a49":
"""

""",
    "q50": "How to find the closest value (to a given scalar) in a vector? (★★☆)",
    "h50": "hint: argmin",
    "a50":
"""

""",
    "q51": "Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)",
    "h51": "hint: dtype",
    "a51":
"""

""",
    "q52": "Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)",
    "h52": "hint: np.atleast_2d, T, np.sqrt",
    "a52":
"""

""",
    "q53": "How to convert a float (32 bits) array into an integer (32 bits) in place?",
    "h53": "hint: astype(copy=False)",
    "a53":
"""

""",
    "q54": """
How to read the following file? (★★☆)
```
1, 2, 3, 4, 5
6,  ,  , 7, 8
 ,  , 9,10,11
```
""",
    "h54": "hint: np.genfromtxt",
    "a54":
"""

""",
    "q55": "What is the equivalent of enumerate for numpy arrays? (★★☆)",
    "h55": "hint: np.ndenumerate, np.ndindex",
    "a55":
"""

""",
    "q56": "Generate a generic 2D Gaussian-like array (★★☆)",
    "h56": "hint: np.meshgrid, np.exp",
    "a56":
"""

""",
    "q57": "How to randomly place p elements in a 2D array? (★★☆)",
    "h57": "hint: np.put, np.random.choice",
    "a57":
"""

""",
    "q58": "Subtract the mean of each row of a matrix (★★☆)",
    "h58": "hint: mean(axis=,keepdims=)",
    "a58":
"""

""",
    "q59": "How to sort an array by the nth column? (★★☆)",
    "h59": "hint: argsort",
    "a59":
"""

""",

    "q60": "How to tell if a given 2D array has null columns? (★★☆)",
    "h60": "hint: any, ~",
    "a60":
"""

""",
    "q61": "Find the nearest value from a given value in an array (★★☆)",
    "h61": "hint: np.abs, argmin, flat",
    "a61":
"""

""",
    "q62": "Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)",
    "h62": "hint: np.nditer",
    "a62":
"""

""",
    "q63": "Create an array class that has a name attribute (★★☆)",
    "h63": "hint: class method",
    "a63":
"""

""",
    "q64": "Consider a given vector, how to add 1 to each element indexed by a second vector "
           "(be careful with repeated indices)? (★★★)",
    "h64": "hint: np.bincount | np.add.at",
    "a64":
"""

""",
    "q65": "How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)",
    "h65": "hint: np.bincount",
    "a65":
"""

""",
    "q66": "Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)",
    "h66": "hint: np.unique",
    "a66":
"""

""",
    "q67": "Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)",
    "h67": "hint: sum(axis=(-2,-1))",
    "a67":
"""

""",
    "q68": "Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S "
           "of same size describing subset  indices? (★★★)",
    "h68": "hint: np.bincount",
    "a68":
"""

""",
    "q69": "How to get the diagonal of a dot product? (★★★)",
    "h69": "hint: np.diag",
    "a69":
"""

""",
    "q70": "Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive "
           "zeros interleaved between each value? (★★★)",
    "h70": "hint: array[::4]",
    "a70":
"""

""",
    "q71": "Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)",
    "h71": "hint: array[:, :, None]",
    "a71":
"""

""",
    "q72": "How to swap two rows of an array? (★★★)",
    "h72": "hint: array[[]] = array[[]]",
    "a72":
"""

""",
    "q73": "Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the "
           "set of unique line segments composing all the  triangles (★★★)",
    "h73": "hint: repeat, np.roll, np.sort, view, np.unique",
    "a73":
"""

""",
    "q74": "Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)",
    "h74": "hint: np.repeat",
    "a74":
"""

""",
    "q75": "How to compute averages using a sliding window over an array? (★★★)",
    "h75": "hint: np.cumsum",
    "a75":
"""

""",
    "q76": "Consider a one-dimensional array Z, build a two-dimensional array whose first row is "
           "(Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be "
           "(Z[-3],Z[-2],Z[-1]) (★★★)",
    "h76": "hint: from numpy.lib import stride_tricks",
    "a76":
"""

""",
    "q77": "How to negate a boolean, or to change the sign of a float inplace? (★★★)",
    "h77": "hint: np.logical_not, np.negative",
    "a77":
"""

""",

    "q78": "Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute "
           "distance from p to each line i (P0[i],P1[i])? (★★★)",
    "h78": "No hints provided...",
    "a78":
"""

""",
    "q79": "Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to "
           "compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)",
    "h79": "No hints provided...",
    "a79":
"""

""",
    "q80": "Consider an arbitrary array, write a function that extract a subpart with a fixed "
           "shape and centered on a given element (pad with a `fill` value when necessary) (★★★)",
    "h80": "hint: minimum maximum",
    "a80":
"""

""",
    "q81": "Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to "
           "generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)",
    "h81": "hint: stride_tricks.as_strided",
    "a81":
"""

""",
    "q82": "Compute a matrix rank (★★★) ",
    "h82": "hint: np.linalg.svd",
    "a82":
"""

""",
    "q83": "How to find the most frequent value in an array?",
    "h83": "hint: np.bincount, argmax",
    "a83":
"""

""",
    "q84": "Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)",
    "h84": "hint: stride_tricks.as_strided",
    "a84":
"""

""",
    "q85": "Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)",
    "h85": "hint: class method",
    "a85":
"""

""",
    "q86": "Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). "
           "How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)",
    "h86": "hint: np.tensordot",
    "a86":
"""

""",
    "q87": "Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)",
    "h87": "hint: np.add.reduceat",
    "a87":
"""

""",

    "q88": "How to implement the Game of Life using numpy arrays? (★★★)",
    "h88": "No hints provided... ",
    "a88":
"""

""",
    "q89": "How to get the n largest values of an array (★★★)",
    "h89": "hint: np.argsort | np.argpartition",
    "a89":
"""

""",
    "q90": "Given an arbitrary number of vectors, build the cartesian product "
           "(every combinations of every item) (★★★)",
    "h90": "hint: np.indices",
    "a90":
"""

""",
    "q91": "How to create a record array from a regular array? (★★★)",
    "h91": "hint: np.core.records.fromarrays",
    "a91":
"""

""",
    "q92": "Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)",
    "h92": "hint: np.power, *, np.einsum",
    "a92":
"""

""",
    "q93": "Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A "
           "that contain elements of each row of B regardless of the order of the elements in B? (★★★)",
    "h93": "hint: np.where",
    "a93":
"""

""",
    "q94": "Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)",
    "h94": "No hints provided...",
    "a94":
"""

""",
    "q95": "Convert a vector of ints into a matrix binary representation (★★★)",
    "h95": "hint: np.unpackbits",
    "a95":
"""

""",
    "q96": "Given a two dimensional array, how to extract unique rows? (★★★)",
    "h96": "hint: np.ascontiguousarray | np.unique",
    "a96":
"""

""",
    "q97": "Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)",
    "h97": "hint: np.einsum",
    "a97":
"""

""",

    "q98": "Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?",
    "h98": "hint: np.cumsum, np.interp ",
    "a98":
"""

""",
    "q99": "Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws "
           "from a multinomial distribution with n degrees, i.e., the rows which only contain integers "
           "and which sum to n. (★★★)",
    "h99": "hint: np.logical_and.reduce, np.mod",
    "a99":
"""

""",
    "q100": "Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., "
            "resample the elements of an array with replacement N times, compute the mean of "
            "each sample, and then compute percentiles over the means). (★★★)",
    "h100": "hint: np.percentile",
    "a100":
"""

""",
}
