===================
100 numpy exercises
===================

A joint effort of the numpy community
-------------------------------------

The goal is both to offer a quick reference for new and old users and to
provide also a set of exercices for those who teach. If you remember having
asked or answered a (short) problem, you can send a pull request. The format
is:

::

  #. Find indices of non-zero elements from [1,2,0,0,4,0]

     .. code:: python

        # Author: Somebody

        print np.nonzero([1,2,0,0,4,0])


Here is what the page looks like so far:
http://http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html

.. Note:: The level names came from an old-game (Dungeon Master)

Repository is at: https://github.com/rougier/numpy-100

The corresponding `IPython notebook
<https://github.com/rougier/numpy-100/blob/master/README.ipynb>`_ is available
from the github repo, thanks to the `rst2ipynb
<https://github.com/esc/rst2ipynb>`_ conversion tool by `Valentin Haenel
<http://haenel.co>`_

Thanks to Michiaki Ariga, there is now a
`Julia version <https://github.com/chezou/julia-100-exercises>`_.


.. **Contents**
.. .. contents::
..     :local:
..     :depth: 1


Neophyte
========

1. Import the numpy package under the name ``np``

   .. code-block:: python

      import numpy as np


2. Print the numpy version and the configuration.

   .. code-block:: python

      print np.__version__
      np.__config__.show()


3. Create a null vector of size 10

   .. code-block:: python

      Z = np.zeros(10)
      print Z


4. Create a null vector of size 10 but the fifth value which is 1

   .. code-block:: python

      Z = np.zeros(10)
      Z[4] = 1
      print Z


5. Create a vector with values ranging from 10 to 49

   .. code-block:: python

      Z = np.arange(10,50)
      print Z


6. Create a 3x3 matrix with values ranging from 0 to 8

   .. code-block:: python

      Z = np.arange(9).reshape(3,3)
      print Z


7. Find indices of non-zero elements from [1,2,0,0,4,0]

   .. code-block:: python

      nz = np.nonzero([1,2,0,0,4,0])
      print nz


8. Create a 3x3 identity matrix

   .. code-block:: python

      Z = np.eye(3)
      print Z


9. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal

   .. code-block:: python

      Z = np.diag(1+np.arange(4),k=-1)
      print Z


10. Create a 3x3x3 array with random values

    .. code-block:: python

       Z = np.random.random((3,3,3))
       print Z


Novice
======

1. Create a 8x8 matrix and fill it with a checkerboard pattern

   .. code-block:: python

      Z = np.zeros((8,8),dtype=int)
      Z[1::2,::2] = 1
      Z[::2,1::2] = 1
      print Z


2. Create a 10x10 array with random values and find the minimum and maximum values

   .. code-block:: python

      Z = np.random.random((10,10))
      Zmin, Zmax = Z.min(), Z.max()
      print Zmin, Zmax


3. Create a checkerboard 8x8 matrix using the tile function

   .. code-block:: python

      Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
      print Z


4. Normalize a 5x5 random matrix (between 0 and 1)

   .. code-block:: python

      Z = np.random.random((5,5))
      Zmax,Zmin = Z.max(), Z.min()
      Z = (Z - Zmin)/(Zmax - Zmin)
      print Z


5. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)

   .. code-block:: python

      Z = np.dot(np.ones((5,3)), np.ones((3,2)))
      print Z


6. Create a 5x5 matrix with row values ranging from 0 to 4

   .. code-block:: python

    Z = np.zeros((5,5))
    Z += np.arange(5)
    print Z


7. Create a vector of size 10 with values ranging from 0 to 1, both excluded

   .. code-block:: python

    Z = np.linspace(0,1,12,endpoint=True)[1:-1]
    print Z


8. Create a random vector of size 10 and sort it

   .. code-block:: python

    Z = np.random.random(10)
    Z.sort()
    print Z


9. Consider two random array A anb B, check if they are equal.

   .. code-block:: python

      A = np.random.randint(0,2,5)
      B = np.random.randint(0,2,5)
      equal = np.allclose(A,B)
      print equal


10. Create a random vector of size 30 and find the mean value

    .. code-block:: python

       Z = np.random.random(30)
       m = Z.mean()
       print m



Apprentice
==========


1. Make an array immutable (read-only)

   .. code-block:: python

      Z = np.zeros(10)
      Z.flags.writeable = False
      Z[0] = 1


2. Consider a random 10x2 matrix representing cartesian coordinates, convert
   them to polar coordinates

   .. code-block:: python

      Z = np.random.random((10,2))
      X,Y = Z[:,0], Z[:,1]
      R = np.sqrt(X**2+Y**2)
      T = np.arctan2(Y,X)
      print R
      print T


3. Create random vector of size 10 and replace the maximum value by 0

   .. code-block:: python

    Z = np.random.random(10)
    Z[Z.argmax()] = 0
    print Z


4. Create a structured array with ``x`` and ``y`` coordinates covering the
   [0,1]x[0,1] area.

   .. code-block:: python

      Z = np.zeros((10,10), [('x',float),('y',float)])
      Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                                   np.linspace(0,1,10))
      print Z


5. Print the minimum and maximum representable value for each numpy scalar type

   .. code-block:: python

      for dtype in [np.int8, np.int32, np.int64]:
         print np.iinfo(dtype).min
         print np.iinfo(dtype).max
      for dtype in [np.float32, np.float64]:
         print np.finfo(dtype).min
         print np.finfo(dtype).max
         print np.finfo(dtype).eps


6. Create a structured array representing a position (x,y) and a color (r,g,b)

   .. code-block:: python

      Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                        ('y', float, 1)]),
                         ('color',    [ ('r', float, 1),
                                        ('g', float, 1),
                                        ('b', float, 1)])])
     print Z


7. Consider a random vector with shape (100,2) representing coordinates, find
   point by point distances

   .. code-block:: python

      Z = np.random.random((10,2))
      X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
      D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
      print D

      # Much faster with scipy
      import scipy
      Z = np.random.random((10,2))
      D = scipy.spatial.distance.cdist(Z,Z)
      print D


8. Generate a generic 2D Gaussian-like array

   .. code-block:: python

      X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
      D = np.sqrt(X*X+Y*Y)
      sigma, mu = 1.0, 0.0
      G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
      print G


9. How to tell if a given 2D array has null columns ?

   .. code-block:: python

      # Author: Warren Weckesser

      Z = np.random.randint(0,3,(3,10))
      print (~Z.any(axis=0)).any()

10. Find the nearest value from a given value in an array

    .. code-block:: python

       Z = np.random.uniform(0,1,10)
       z = 0.5
       m = Z.flat[np.abs(Z - z).argmin()]
       print m


Journeyman
==========

1. Consider the following file::

    1,2,3,4,5
    6,,,7,8
    ,,9,10,11

   How to read it ?

   .. code-block:: python

      Z = np.genfromtxt("missing.dat", delimiter=",")


2. Consider a generator function that generates 10 integers and use it to build an
   array

   .. code-block:: python

      def generate():
          for x in xrange(10):
              yield x
      Z = np.fromiter(generate(),dtype=float,count=-1)
      print Z


3. Consider a given vector, how to add 1 to each element indexed by a second
   vector (be careful with repeated indices) ?

   .. code-block:: python

      # Author: Brett Olsen

      Z = np.ones(10)
      I = np.random.randint(0,len(Z),20)
      Z += np.bincount(I, minlength=len(Z))
      print Z


4. How to accumulate elements of a vector (X) to an array (F) based on an index
   list (I) ?

   .. code-block:: python

      # Author: Alan G Isaac

      X = [1,2,3,4,5,6]
      I = [1,3,9,3,4,1]
      F = np.bincount(I,X)
      print F


5. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique
   colors

   .. code-block:: python

      # Author: Nadav Horesh

      w,h = 16,16
      I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
      F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
      n = len(np.unique(F))
      print np.unique(I)


6. Considering a four dimensions array, how to get sum over the last two axis at once ?

   .. code-block:: python

      A = np.random.randint(0,10,(3,4,3,4))
      sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
      print


7. Considering a one-dimensional vector D, how to compute means of subsets of D
   using a vector S of same size describing subset indices ?


   .. code-block:: python

      # Author: Jaime Fernández del Río

      D = np.random.uniform(0,1,100)
      S = np.random.randint(0,10,100)
      D_sums = np.bincount(S, weights=D)
      D_counts = np.bincount(S)
      D_means = D_sums / D_counts
      print D_means


8. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3
   consecutive zeros interleaved between each value ?

   .. code-block:: python

      # Author: Warren Weckesser

      Z = np.array([1,2,3,4,5])
      nz = 3
      Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
      Z0[::nz+1] = Z
      print Z0


9. Consider an array of dimension (5,5,3), how to mulitply it by an array with
   dimensions (5,5) ?

   .. code-block:: python

      A = np.ones((5,5,3))
      B = 2*np.ones((5,5))
      print A * B[:,:,None]


10. How to swap two rows of an array ?


    .. code-block:: python

       # Author: Eelco Hoogendoorn

       A = np.arange(25).reshape(5,5)
       A[[0,1]] = A[[1,0]]
       print A


Craftsman
=========

1. Consider a one-dimensional array Z, build a two-dimensional array whose
   first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last
   row should be (Z[-3],Z[-2],Z[-1])

   .. code-block:: python

      # Author: Joe Kington / Erik Rigtorp
      from numpy.lib import stride_tricks

      def rolling(a, window):
          shape = (a.size - window + 1, window)
          strides = (a.itemsize, a.itemsize)
          return stride_tricks.as_strided(a, shape=shape, strides=strides)
      Z = rolling(np.arange(10), 3)
      print Z


2. Consider a set of 10 triplets describing 10 triangles (with shared
   vertices), find the set of unique line segments composing all the triangles.

   .. code-block:: python

      # Author: Nicolas P. Rougier

      faces = np.random.randint(0,100,(10,3))
      F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
      F = F.reshape(len(F)*3,2)
      F = np.sort(F,axis=1)
      G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
      G = np.unique(G)
      print G


3. Given an array C that is a bincount, how to produce an array A such that
   np.bincount(A) == C ?

   .. code-block:: python

     # Author: Jaime Fernández del Río

     C = np.bincount([1,1,2,3,4,4,6])
     A = np.repeat(np.arange(len(C)), C)
     print A

4. How to compute averages using a sliding window over an array ?

   .. code-block:: python

      # Author: Jaime Fernández del Río

      def moving_average(a, n=3) :
          ret = np.cumsum(a, dtype=float)
          ret[n:] = ret[n:] - ret[:-n]
          return ret[n - 1:] / n
      Z = np.arange(20)
      print moving_average(Z, n=3)

5. How to get the documentation of the numpy add function from the command line ?

   .. code-block:: bash

      python -c "import numpy; numpy.info(numpy.add)"

6. How to negate a boolean, or to change the sign of a float inplace ?

  .. code-block:: python

     # Author: Nathaniel J. Smith

     Z = np.random.randint(0,2,100)
     np.logical_not(arr, out=arr)

     Z = np.random.uniform(-1.0,1.0,100)
     np.negative(arr, out=arr)

7.

Artisan
=======

1. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])

   .. code-block:: python

      # Author: Robert Kern

      Z = np.random.randint(0,5,(10,3))
      E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
      U = Z[~E]
      print Z
      print U

2. Convert a vector of ints into a matrix binary representation.

   .. code-block:: python

      # Author: Warren Weckesser

      I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
      B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
      print B[:,::-1]

      # Author: Daniel T. McDonald

      I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
      print np.unpackbits(I[:, np.newaxis], axis=1)


3. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to
   compute distance from p to each line i (P0[i],P1[i]) ?

   .. code-block:: python

      def distance(P0, P1, p):
          T = P1 - P0
          L = (T**2).sum(axis=1)
          U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
          U = U.reshape(len(U),1)
          D = P0 + U*T - p
          return np.sqrt((D**2).sum(axis=1))

      P0 = np.random.uniform(-10,10,(10,2))
      P1 = np.random.uniform(-10,10,(10,2))
      p  = np.random.uniform(-10,10,( 1,2))
      print distance(P0, P1, p)


4. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P,
   how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i]) ?

   .. code-block:: python

      Answer needed actually



Adept
=====

1. Consider an arbitrary array, write a function that extract a subpart with a
   fixed shape and centered on a given element (pad with a ``fill`` value when
   necessary)

   .. code:: python

      # Author: Nicolas Rougier

      Z = np.random.randint(0,10,(10,10))
      shape = (5,5)
      fill  = 0
      position = (1,1)

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
      print Z
      print R


2. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an
   array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] ?

   .. code-block:: python

      # Author: Stéfan van der Walt

      Z = np.arange(1,15,dtype=uint32)
      R = stride_tricks.as_strided(Z,(11,4),(4,4))
      print R


Expert
======

1. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A
   that contain elements of each row of B regardless of the order of the elements
   in B ?

   .. code-block:: python

      # Author: Gabe Schwartz

      A = np.random.randint(0,5,(8,3))
      B = np.random.randint(0,5,(2,2))

      C = (A[..., np.newaxis, np.newaxis] == B)
      rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
      print rows


2. Extract all the contiguous 3x3 blocks from a random 10x10 matrix.

   .. code-block:: python

      # Author: Chris Barker

      Z = np.random.randint(0,5,(10,10))
      n = 3
      i = 1 + (Z.shape[0]-3)
      j = 1 + (Z.shape[1]-3)
      C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
      print C


3. Create a 2D array subclass such that Z[i,j] == Z[j,i]

   .. code-block:: python

      # Author: Eric O. Lebigot
      # Note: only works for 2d array and value setting using indices

      class Symetric(np.ndarray):
          def __setitem__(self, (i,j), value):
              super(Symetric, self).__setitem__((i,j), value)
              super(Symetric, self).__setitem__((j,i), value)

      def symetric(Z):
          return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

      S = symetric(np.random.randint(0,10,(5,5)))
      S[2,3] = 42
      print S

4. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1).
   How to compute the sum of of the p matrix products at once ? (result has shape (n,1))

   .. code-block:: python

      # Author: Stéfan van der Walt

      p, n = 10, 20
      M = np.ones((p,n,n))
      V = np.ones((p,n,1))
      S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
      print S

      # It works, because:
      # M is (p,n,n)
      # V is (p,n,1)
      # Thus, summing over the paired axes 0 and 0 (of M and V independently),
      # and 2 and 1, to remain with a (n,1) vector.


Master
======

1. Given a two dimensional array, how to extract unique rows ?

   .. note:: See `stackoverflow <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/>`_ for explanations.

   .. code-block:: python

      # Author: Jaime Fernández del Río

      Z = np.random.randint(0,2,(6,3))
      T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
      _, idx = np.unique(T, return_index=True)
      uZ = Z[idx]
      print uZ

2. How to implement the Game of Life using numpy arrays ?

   .. code-block:: python

      # Author: Nicolas Rougier

      def iterate(Z):
          # Count neighbours
          N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
               Z[1:-1,0:-2]                + Z[1:-1,2:] +
               Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

          # Apply rules
          birth = (N==3) & (Z[1:-1,1:-1]==0)
          survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
          Z[...] = 0
          Z[1:-1,1:-1][birth | survive] = 1
          return Z

      Z = np.random.randint(0,2,(50,50))
      for i in range(100): Z = iterate(Z)



Archmaster
==========
