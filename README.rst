===================
100 numpy exercises
===================

A joint effort of the numpy community
-------------------------------------

The goal is both to offer a quick reference for new and old users and to
provide also a set of exercices for those who teach. If you remember having
asked or answered a (short) problem, you can send a pull request. The format
is::

 #. Find indices of non-zero elements from [1,2,0,0,4,0]

    .. code:: python

       # Author: Somebody

       print np.nonzero([1,2,0,0,4,0])


Here is what the page looks like so far:
http://www.loria.fr/~rougier/teaching/numpy.100/index.html


.. Note:: The level names came from an old-game (Dungeon Master)


Repository is at: https://github.com/rougier/numpy-100


**Contents**

.. contents::
   :local:
   :depth: 1


Neophyte
========

1. Import the numpy package under the name ``np``

   .. code:: python

      import numpy as np


2. Print the numpy version and the configuration.

   .. code:: python

      print np.__version__
      np.__config__.show()


3. Create a null vector of size 10

   .. code:: python

      Z = np.zeros(10)

4. Create a null vector of size 10 but the fifth value which is 1

   .. code:: python

      Z = np.zeros(10)
      Z[4] = 1

5. Create a vector with values ranging from 10 to 99

   .. code:: python

      Z = np.arange(10,100)

6. Create a 3x3 matrix with values ranging from 0 to 8

   .. code:: python

      Z = np.arange(9).reshape(3,3)

7. Find indices of non-zero elements from [1,2,0,0,4,0]

   .. code:: python

      nz = np.nonzero([1,2,0,0,4,0])


8. Create a 3x3 identity matrix

   .. code:: python

      Z = np.eye(3)

9. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal

   .. code:: python

      Z = np.diag(1+np.arange(4),k=-1)


10. Create a 10x10x10 array with random values

    .. code:: python

       Z = np.random.random((10,10,10))

Novice
======

1. Create a 8x8 matrix and fill it with a checkerboard pattern

   .. code:: python

      Z = np.zeros((8,8))
      Z[1::2,::2] = 1
      Z[::2,1::2] = 1

2. Create a 10x10 array with random values and find the minimum and maximum values

   .. code:: python

      Z = np.random.random((10,10))
      Zmin, Zmax = Z.min(), Z.max()

3. Create a checkerboard 8x8 matrix using the tile function

   .. code:: python

      Z = np.tile( np.array([[0,1],[1,0]]), (4,4))

4. Normalize a 5x5 random matrix (between 0 and 1)

   .. code:: python

      Z = np.random.random((5,5))
      Zmax,Zmin = Z.max(), Z.min()
      Z = (Z - Zmin)/(Zmax - Zmin)


5. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)

   .. code:: python

      Z = np.dot(np.ones((5,3)), np.ones((3,2)))


6. Create a 10x10 matrix with row values ranging from 0 to 9

   .. code:: python

    Z = np.zeros((10,10))
    Z += np.arange(10)

7. Create a vector of size 1000 with values ranging from 0 to 1, both excluded

   .. code:: python

    Z = np.random.linspace(0,1,1002,endpoint=True)[1:-1]

8. Create a random vector of size 100 and sort it

   .. code:: python

    Z = np.random.random(100)
    Z.sort()

9. Consider two random matrices A anb B, check if they are equal.

   .. code:: python

      A = np.random.randint(0,2,(2,2))
      B = np.random.randint(0,2,(2,2))
      equal = np.allclose(A,B)

10. Create a random vector of size 1000 and find the mean value

    .. code:: python

       Z = np.random.random(1000)
       m = Z.mean()



Apprentice
==========


1. Make an array immutable (read-only)

   .. code:: python

      Z = np.zeros(10)
      Z.flags.writeable = False


2. Consider a random 100x2 matrix representing cartesian coordinates, convert
   them to polar coordinates

   .. code:: python

      Z = np.random.random((100,2))
      X,Y = Z[:,0], Z[:,1]
      R = np.sqrt(X**2+Y**2)
      T = np.arctan2(Y,X)


3. Create random vector of size 100 and replace the maximum value by 0

   .. code:: python

    Z = np.random.random(100)
    Z[Z.argmax()] = 0


4. Create a structured array with ``x`` and ``y`` coordinates covering the
   [0,1]x[0,1] area.

   .. code:: python

      Z = np.zeros((10,10), [('x',float),('y',float)])
      Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                                   np.linspace(0,1,10))

5. Print the minimum and maximum representable value for each numpy scalar type

   .. code:: python

      for dtype in [np.int8, np.int32, np.int64]:
         print np.iinfo(dtype).min
         print np.iinfo(dtype).max
      for dtype in [np.float32, np.float64]:
         print np.finfo(dtype).min
         print np.finfo(dtype).max
         print np.finfo(dtype).eps


6. Create a structured array representing a position (x,y) and a color (r,g,b)

   .. code:: python

      Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                        ('y', float, 1)]),
                         ('color',    [ ('r', float, 1),
                                        ('g', float, 1),
                                        ('b', float, 1)])])


7. Consider a random vector with shape (100,2) representing coordinates, find
   point by point distances

   .. code:: python

      Z = np.random.random((10,2))
      X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
      D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)

      # Much faster with scipy
      Z = np.random.random((10,2))
      D = scipy.spatial.distance.cdist(Z,Z)



8. Generate a generic 2D Gaussian-like array

   .. code:: python

      X, Y = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
      D = np.sqrt(X*X+Y*Y)
      sigma, mu = 1.0, 0.0
      G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )

9. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3
   consecutive zeros interleaved between each value ?

   .. code:: python

      # Author: Warren Weckesser

      Z = np.array([1,2,3,4,5])
      nz = 3
      Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
      Z0[::nz+1] = Z


10. Find the nearest value from a given value in an array

    .. code:: python

       Z.flat[np.abs(Z - z).argmin()]



Journeyman
==========

1. Consider the following file::

    1,2,3,4,5
    6,,,7,8
    ,,9,10,11

   How to read it ?

   .. code:: python

      Z = genfromtxt("missing.dat", delimiter=",")


2. Consider a generator function that generates 10 integers and use it to build an
   array

   .. code:: python

      def generate():
          for x in xrange(10):
              yield x
      Z = np.fromiter(generate(),dtype=float,count=-1)


3. Consider a given vector, how to add 1 to each element indexed by a second
   vector (be careful with repeated indices) ?

   .. code:: python

      # Author: Brett Olsen

      Z = np.ones(10)
      I = np.random.randint(0,len(Z),20)
      Z += np.bincount(I, minlength=len(Z))


4. How to accumulate elements of a vector (X) to an array (F) based on an index
   list (I) ?

   .. code:: python

      # Author: Alan G Isaac

      X = [1,2,3,4,5,6]
      I = [1,3,9,3,4,1]
      F = np.bincount(I,X)

5. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique
   colors

   .. code:: python

      # Author: Nadav Horesh

      w,h = 16,16
      I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
      F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
      n = len(np.unique(F))

      np.unique(I)

6. Considering a four dimensions array, how to get sum over the last two axis at once ?


   .. code:: python

      A = np.random.randint(0,10,(3,4,3,4))
      sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)


7. Considering a one-dimensional vector D, how to compute means of subsets of D
   using a vector S of same size describing subset indices ?


   .. code:: python

      # Jaime Fernández del Río

      D = np.random.uniform(0,1,100)
      S = np.random.randint(0,10,100)
      D_sums = np.bincount(S, weights=D)
      D_counts = np.bincount(S)
      D_means = D_sums / D_counts





Craftsman
=========

1. Consider a one-dimensional array Z, build a two-dimensional array whose
   first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last
   row should be (Z[-3],Z[-2],Z[-1])

   .. code:: python

      # Author: Joe Kington / Erik Rigtorp

      def rolling(a, window):
          shape = (a.size - window + 1, window)
          strides = (a.itemsize, a.itemsize)
          return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

      Z = rolling(np.arange(100), 3)

2. Consider a set of 100 triplets describing 100 triangles (with shared
   vertices), find the set of unique line segments composing all the triangles.

   .. code:: python

      # Author: Nicolas Rougier

      faces = np.random.randint(0,100,(100,3))

      F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
      F = F.reshape(len(F)*3,2)
      F = np.sort(F,axis=1)
      G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
      G = np.unique(G)


3. Given an array C that is a bincount, how to produce an array A such that
   np.bincount(A) == C ?

   .. code:: python

     # Jaime Fernández del Río

     C = np.bincount([1,1,2,3,4,4,6])
     A = np.repeat(np.arange(len(C)), C)



Artisan
=======

1. Considering a 100x3 matrix, extract rows with unequal values (e.g. [2,2,3])

   .. code:: python

      # Author: Robert Kern

      Z = np.random.randint(0,5,(100,3))
      E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
      U = Z[~E]

2. Convert a vector of ints into a matrix binary representation.

   .. code:: python

      # Author: Warren Weckesser

      I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
      B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
      B = B[:,::-1]

      # Author: Daniel T. McDonald

      I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
      np.unpackbits(I[:, np.newaxis], axis=1)



Adept
=====

1. Consider an arbitrary array, write a function that extract a subpart with a
   fixed shape and centered on a given element (pad with a ``fill`` value when
   necessary)

   .. code :: python

      # Author: Nicolas Rougier

      Z = np.random.random((25,25))
      shape = (3,3)
      fill  = 0
      position = (0,0)

      R = np.ones(shape, dtype=Z.dtype)*fill
      P  = np.array(list(position)).astype(int)
      Rs = np.array(list(R.shape)).astype(int)
      Zs = np.array(list(Z.shape)).astype(int)

      R_start = np.zeros((len(shape),)).astype(int)
      R_stop  = np.array(list(shape)).astype(int)
      Z_start = (P-Rs//2)
      Z_stop  = (P+Rs//2)+Rs%2

      R_start = (R_start - np.minimum(Z_start,0)).tolist()
      Z_start = (np.maximum(Z_start,0)).tolist()
      R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
      Z_stop = (np.minimum(Z_stop,Zs)).tolist()

      r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
      z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
      R[r] = Z[z]


2. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an
   array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] ?

   .. code:: python

      # Stéfan van der Walt

      Z = np.arange(1,15)
      R = as_strided(Z,(11,4),(4,4))





Expert
======

1. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A
   that contain elements of each row of B regardless of the order of the elements
   in B ?

   .. code:: python

      # Author: Gabe Schwartz

      A = np.random.randint(0,5,(8,3))
      B = np.random.randint(0,5,(2,2))

      C = (A[..., np.newaxis, np.newaxis] == B)
      rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]


2. Extract all the contiguous 3x3 blocks from a random 10x10 matrix.

   .. code:: python

      # Chris Barker

      Z = np.random.randint(0,5,(10,10))
      n = 3
      i = 1 + (Z.shape[0]-3)
      j = 1 + (Z.shape[1]-3)
      C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)


3. Create a 2D array subclass such that Z[i,j] == Z[j,i]

   .. code:: python

      # Eric O. Lebigot
      # Note: only works for 2d array and value setting using indices

      class Symetric(np.ndarray):
          def __setitem__(self, (i,j), value):
              super(Symetric, self).__setitem__((i,j), value)
              super(Symetric, self).__setitem__((j,i), value)

      def symetric(Z):
          return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

      S = symetric(np.random.randint(0,10,(5,5))
      S[2,3] = 42
      print S


Master
======

1. Given a two dimensional array, how to extract unique rows ?

   .. note:: See `stackoverflow <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/>`_ for explanations.

   .. code:: python

      # Jaime Fernández del Río

      Z = np.random.randint(0,2,(6,6))
      T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
      _, idx = np.unique(T, return_index=True)
      uZ = Z[idx]



Archmaster
==========
