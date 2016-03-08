100 numpy exercises
=====================================

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

        print(np.nonzero([1,2,0,0,4,0]))


Here is what the page looks like so far:
http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html

Repository is at: https://github.com/rougier/numpy-100

Thanks to Michiaki Ariga, there is now a
`Julia version <https://github.com/chezou/julia-100-exercises>`_.


#. Import the numpy package under the name ``np`` (★☆☆) 

   .. code-block:: python

      import numpy as np


#. Print the numpy version and the configuration (★☆☆) 

   .. code-block:: python

      print(np.__version__)
      np.__config__.show()


#. Create a null vector of size 10 (★☆☆) 

   .. code-block:: python

      Z = np.zeros(10)
      print(Z)

      
#. How to get the documentation of the numpy add function from the command line ? (★☆☆) 

   .. code-block:: bash

      python -c "import numpy; numpy.info(numpy.add)"


#. Create a null vector of size 10 but the fifth value which is 1 (★☆☆) 

   .. code-block:: python

      Z = np.zeros(10)
      Z[4] = 1
      print(Z)


#. Create a vector with values ranging from 10 to 49 (★☆☆) 

   .. code-block:: python

      Z = np.arange(10,50)
      print(Z)


#. Reverse a vector (first element becomes last) (★☆☆) 

   .. code-block:: python

      Z = np.arange(50)
      Z = Z[::-1]


#. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆) 

   .. code-block:: python

      Z = np.arange(9).reshape(3,3)
      print(Z)


#. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆) 

   .. code-block:: python

      nz = np.nonzero([1,2,0,0,4,0])
      print(nz)


#. Create a 3x3 identity matrix (★☆☆) 

   .. code-block:: python

      Z = np.eye(3)
      print(Z)


#. Create a 3x3x3 array with random values (★☆☆) 

   .. code-block:: python

      Z = np.random.random((3,3,3))
      print(Z)


#. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆) 

   .. code-block:: python

      Z = np.random.random((10,10))
      Zmin, Zmax = Z.min(), Z.max()
      print(Zmin, Zmax)

      
#. Create a random vector of size 30 and find the mean value  (★☆☆) 

   .. code-block:: python

      Z = np.random.random(30)
      m = Z.mean()
      print(m)

      
#. Create a 2d array with 1 on the border and 0 inside  (★☆☆) 

   .. code-block:: python

      Z = np.ones((10,10))
      Z[1:-1,1:-1] = 0

                   
#. What is the result of the following expression ? (★☆☆)

   .. code-block:: python

      0 * np.nan
      np.nan == np.nan
      np.inf > np.nan
      np.nan - np.nan
      0.3 == 3 * 0.1

      
#. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆) 

   .. code-block:: python

      Z = np.diag(1+np.arange(4),k=-1)
      print(Z)
      

#. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆) 

   .. code-block:: python

      Z = np.zeros((8,8),dtype=int)
      Z[1::2,::2] = 1
      Z[::2,1::2] = 1
      print(Z)

#. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element ?

   .. code-block:: python

      print(np.unravel_index(100,(6,7,8)))

#. Create a checkerboard 8x8 matrix using the tile function (★☆☆) 

   .. code-block:: python

      Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
      print(Z)


#. Normalize a 5x5 random matrix (★☆☆) 

   .. code-block:: python

      Z = np.random.random((5,5))
      Zmax, Zmin = Z.max(), Z.min()
      Z = (Z - Zmin)/(Zmax - Zmin)
      print(Z)


#. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆) 

   .. code-block:: python

      Z = np.dot(np.ones((5,3)), np.ones((3,2)))
      print(Z)


#. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆) 

   .. code-block:: python

    Z = np.zeros((5,5))
    Z += np.arange(5)
    print(Z)

    
#. Consider a generator function that generates 10 integers and use it to build an
   array (★☆☆) 

   .. code-block:: python

      def generate():
          for x in xrange(10):
              yield x
      Z = np.fromiter(generate(),dtype=float,count=-1)
      print(Z)


#. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆) 

   .. code-block:: python

    Z = np.linspace(0,1,12,endpoint=True)[1:-1]
    print(Z)


#. Create a random vector of size 10 and sort it  (★★☆) 

   .. code-block:: python

    Z = np.random.random(10)
    Z.sort()
    print(Z)


#. Consider two random array A anb B, check if they are equal  (★★☆) 

   .. code-block:: python

      A = np.random.randint(0,2,5)
      B = np.random.randint(0,2,5)
      equal = np.allclose(A,B)
      print(equal)



#. Make an array immutable (read-only) (★★☆) 

   .. code-block:: python

      Z = np.zeros(10)
      Z.flags.writeable = False
      Z[0] = 1


#. Consider a random 10x2 matrix representing cartesian coordinates, convert
   them to polar coordinates (★★☆) 

   .. code-block:: python

      Z = np.random.random((10,2))
      X,Y = Z[:,0], Z[:,1]
      R = np.sqrt(X**2+Y**2)
      T = np.arctan2(Y,X)
      print(R)
      print(T)


#. Create random vector of size 10 and replace the maximum value by 0 (★★☆) 

   .. code-block:: python

    Z = np.random.random(10)
    Z[Z.argmax()] = 0
    print(Z)


#. Create a structured array with ``x`` and ``y`` coordinates covering the
   [0,1]x[0,1] area (★★☆) 

   .. code-block:: python

      Z = np.zeros((10,10), [('x',float),('y',float)])
      Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
                                   np.linspace(0,1,10))
      print(Z)


#. Print the minimum and maximum representable value for each numpy scalar type (★★☆) 

   .. code-block:: python

      for dtype in [np.int8, np.int32, np.int64]:
         print(np.iinfo(dtype).min)
         print(np.iinfo(dtype).max)
      for dtype in [np.float32, np.float64]:
         print(np.finfo(dtype).min)
         print(np.finfo(dtype).max)
         print(np.finfo(dtype).eps)

#. How to print all the values of an array ?  (★★☆)

   .. code-block:: python

      np.set_printoptions(threshold=np.nan)
      Z = np.zeros((25,25))
      print(Z)

#. How to print all the values of an array ?  (★★☆)

   .. code-block:: python

      np.set_printoptions(threshold=np.nan)
      Z = np.zeros((25,25))
      print(Z)

#. How to find the closest value (to a given scalar) in an array ?  (★★☆)
   
   .. code-block:: python

      Z = np.arange(100)
      v = np.random.uniform(0,100)
      index = (np.abs(Z-v)).argmin()
      print(Z[index])

#. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

   .. code-block:: python

      Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                        ('y', float, 1)]),
                         ('color',    [ ('r', float, 1),
                                        ('g', float, 1),
                                        ('b', float, 1)])])
     print(Z)


#. Consider a random vector with shape (100,2) representing coordinates, find
   point by point distances (★★☆) 

   .. code-block:: python

      Z = np.random.random((10,2))
      X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
      D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
      print(D)

      # Much faster with scipy
      import scipy
      # Thanks Gavin Heverly-Coulson (#issue 1)
      import scipy.spatial
      
      Z = np.random.random((10,2))
      D = scipy.spatial.distance.cdist(Z,Z)
      print(D)


#. How to convert a float (32 bits) array into an integer (32 bits) in place ?

   .. code-block:: python

      Z = np.arange(10, dtype=np.int32)
      Z = Z.astype(np.float32, copy=False)
                   
   
#. Consider the following file::

    1,2,3,4,5
    6,,,7,8
    ,,9,10,11

   How to read it ? (★★☆) 

   .. code-block:: python

      Z = np.genfromtxt("missing.dat", delimiter=",")


#. What is the equivalent of `enumerate` for numpy arrays ? (★★☆)

   .. code-block:: python

      Z = np.arange(9).reshape(3,3)
      for index, value in np.ndenumerate(Z):
          print(index, value)
      for index in np.ndindex(Z.shape):
          print(index, Z[index])
                   
         
#. Generate a generic 2D Gaussian-like array (★★☆) 

   .. code-block:: python

      X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
      D = np.sqrt(X*X+Y*Y)
      sigma, mu = 1.0, 0.0
      G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
      print(G)

#. How to randomly place p elements in a 2D array ? (★★☆)

   .. code-block:: python

      # Author: Divakar
       
      n = 10
      p = 3
      Z = np.zeros((n,n))
      np.put(Z, np.random.choice(range(n*n), p, replace=False),1)

#. Subtract the mean of each row of a matrix (★★☆) 

   .. code-block:: python

      # Author: Warren Weckesser

      X = np.random.rand(5, 10)

      # Recent versions of numpy
      Y = X - X.mean(axis=1, keepdims=True)

      # Older versions of numpy
      Y = X - X.mean(axis=1).reshape(-1, 1)

#. How to I sort an array by the nth column ? (★★☆) 

   .. code-block:: python

      # Author: Steve Tjoa
      
      Z = np.random.randint(0,10,(3,3))
      print(Z)
      print(Z[Z[:,1].argsort()])
   

#. How to tell if a given 2D array has null columns ? (★★☆) 

   .. code-block:: python

      # Author: Warren Weckesser

      Z = np.random.randint(0,3,(3,10))
      print((~Z.any(axis=0)).any())

      
#. Find the nearest value from a given value in an array (★★☆) 

   .. code-block:: python

      Z = np.random.uniform(0,1,10)
      z = 0.5
      m = Z.flat[np.abs(Z - z).argmin()]
      print(m)





#. Consider a given vector, how to add 1 to each element indexed by a second
   vector (be careful with repeated indices) ? (★★★) 

   .. code-block:: python

      # Author: Brett Olsen

      Z = np.ones(10)
      I = np.random.randint(0,len(Z),20)
      Z += np.bincount(I, minlength=len(Z))
      print(Z)


#. How to accumulate elements of a vector (X) to an array (F) based on an index
   list (I) ? (★★★) 

   .. code-block:: python

      # Author: Alan G Isaac

      X = [1,2,3,4,5,6]
      I = [1,3,9,3,4,1]
      F = np.bincount(I,X)
      print(F)


#. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique
   colors (★★★) 

   .. code-block:: python

      # Author: Nadav Horesh

      w,h = 16,16
      I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
      F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
      n = len(np.unique(F))
      print(np.unique(I))


#. Considering a four dimensions array, how to get sum over the last two axis
   at once ? (★★★)

   .. code-block:: python

      A = np.random.randint(0,10,(3,4,3,4))
      sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
      print(sum)


#. Considering a one-dimensional vector D, how to compute means of subsets of D
   using a vector S of same size describing subset indices ? (★★★) 

   .. code-block:: python

      # Author: Jaime Fernández del Río

      D = np.random.uniform(0,1,100)
      S = np.random.randint(0,10,100)
      D_sums = np.bincount(S, weights=D)
      D_counts = np.bincount(S)
      D_means = D_sums / D_counts
      print(D_means)


#. How to get the diagonal of a dot product ? (★★★) 

   .. code-block:: python

      # Author: Mathieu Blondel
                   
      # Slow version  
      np.diag(np.dot(A, B))

      # Fast version
      np.sum(A * B.T, axis=1)

      # Faster version
      np.einsum("ij,ji->i", A, B).


#. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3
   consecutive zeros interleaved between each value ?  (★★★) 

   .. code-block:: python

      # Author: Warren Weckesser

      Z = np.array([1,2,3,4,5])
      nz = 3
      Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
      Z0[::nz+1] = Z
      print(Z0)


#. Consider an array of dimension (5,5,3), how to mulitply it by an array with
   dimensions (5,5) ?  (★★★) 

   .. code-block:: python

      A = np.ones((5,5,3))
      B = 2*np.ones((5,5))
      print(A * B[:,:,None])


#. How to swap two rows of an array ? (★★★) 

   .. code-block:: python

      # Author: Eelco Hoogendoorn

      A = np.arange(25).reshape(5,5)
      A[[0,1]] = A[[1,0]]
      print(A)


#. Consider a set of 10 triplets describing 10 triangles (with shared
   vertices), find the set of unique line segments composing all the triangles (★★★) 

   .. code-block:: python

      # Author: Nicolas P. Rougier

      faces = np.random.randint(0,100,(10,3))
      F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
      F = F.reshape(len(F)*3,2)
      F = np.sort(F,axis=1)
      G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
      G = np.unique(G)
      print(G)


#. Given an array C that is a bincount, how to produce an array A such that
   np.bincount(A) == C ?  (★★★) 

   .. code-block:: python

     # Author: Jaime Fernández del Río

     C = np.bincount([1,1,2,3,4,4,6])
     A = np.repeat(np.arange(len(C)), C)
     print(A)

     
#. How to compute averages using a sliding window over an array ? (★★★) 

   .. code-block:: python

      # Author: Jaime Fernández del Río

      def moving_average(a, n=3) :
          ret = np.cumsum(a, dtype=float)
          ret[n:] = ret[n:] - ret[:-n]
          return ret[n - 1:] / n
      Z = np.arange(20)
      print(moving_average(Z, n=3))

#. Consider a one-dimensional array Z, build a two-dimensional array whose
   first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last
   row should be (Z[-3],Z[-2],Z[-1]) (★★★) 

   .. code-block:: python

      # Author: Joe Kington / Erik Rigtorp
      from numpy.lib import stride_tricks

      def rolling(a, window):
          shape = (a.size - window + 1, window)
          strides = (a.itemsize, a.itemsize)
          return stride_tricks.as_strided(a, shape=shape, strides=strides)
      Z = rolling(np.arange(10), 3)
      print(Z)


#. How to negate a boolean, or to change the sign of a float inplace ? (★★★) 

   .. code-block:: python

      # Author: Nathaniel J. Smith

      Z = np.random.randint(0,2,100)
      np.logical_not(arr, out=arr)

      Z = np.random.uniform(-1.0,1.0,100)
      np.negative(arr, out=arr)



#. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to
   compute distance from p to each line i (P0[i],P1[i]) ? (★★★) 

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
      print(distance(P0, P1, p))


#. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P,
   how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i]) ? (★★★) 

   .. code-block:: python

      # Author: Italmassov Kuanysh
      # based on distance function from previous question
      P0 = np.random.uniform(-10, 10, (10,2))
      P1 = np.random.uniform(-10,10,(10,2))
      p = np.random.uniform(-10, 10, (10,2))
      print np.array([distance(P0,P1,p_i) for p_i in p])

#. Consider an arbitrary array, write a function that extract a subpart with a
   fixed shape and centered on a given element (pad with a ``fill`` value when
   necessary)  (★★★) 

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
      print(Z)
      print(R)


#. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an
   array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] ? (★★★) 

   .. code-block:: python

      # Author: Stefan van der Walt

      Z = np.arange(1,15,dtype=uint32)
      R = stride_tricks.as_strided(Z,(11,4),(4,4))
      print(R)

#. Compute a matrix rank (★★★)

   .. code-block:: python

      # Author: Stefan van der Walt
    
      Z = np.random.uniform(0,1,(10,10))
      U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
      rank = np.sum(S > 1e-10)

#. How to find the most frequent value in an array ?

   .. code-block:: python

      Z = np.random.randint(0,10,50)
      print(np.bincount(Z).argmax())
      
#. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★) 

   .. code-block:: python

      # Author: Chris Barker

      Z = np.random.randint(0,5,(10,10))
      n = 3
      i = 1 + (Z.shape[0]-3)
      j = 1 + (Z.shape[1]-3)
      C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
      print(C)


#. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★) 

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
      print(S)

#. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1).
   How to compute the sum of of the p matrix products at once ? (result has shape (n,1)) (★★★) 

   .. code-block:: python

      # Author: Stefan van der Walt

      p, n = 10, 20
      M = np.ones((p,n,n))
      V = np.ones((p,n,1))
      S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
      print(S)

      # It works, because:
      # M is (p,n,n)
      # V is (p,n,1)
      # Thus, summing over the paired axes 0 and 0 (of M and V independently),
      # and 2 and 1, to remain with a (n,1) vector.


#. Consider a 16x16 array, how to get the block-sum (block size is 4x4) ? (★★★) 

   .. code-block:: python

      # Author: Robert Kern
                         
      Z = np.ones(16,16)
      k = 4
      S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                             np.arange(0, Z.shape[1], k), axis=1)
                   

#. How to implement the Game of Life using numpy arrays ? (★★★) 

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

#. How to get the n largest values of an array (★★★)

   .. code-block:: python

      Z = np.arange(10000)
      np.random.shuffle(Z)
      n = 5

      # Slow
      print (Z[np.argsort(Z)[-n:]])
      
      # Fast
      print (Z[np.argpartition(-Z,n)[:n]])
      

#. Given an arbitrary number of vectors, build the cartesian product (every
   combinations of every item) (★★★)
   
   .. code-block:: python

      # Author: Stefan Van der Walt

      def cartesian(arrays):
          arrays = [np.asarray(a) for a in arrays]
          shape = (len(x) for x in arrays)

          ix = np.indices(shape, dtype=int)
          ix = ix.reshape(len(arrays), -1).T

          for n, arr in enumerate(arrays):
              ix[:, n] = arrays[n][ix[:, n]]

          return ix

      print (cartesian(([1, 2, 3], [4, 5], [6, 7])))


#. How to create a record array from a regular array ? (★★★) 
   
   .. code-block:: python

      Z = np.array([("Hello", 2.5, 3),
                    ("World", 3.6, 2)])
      R = np.core.records.fromarrays(Z.T, 
                                     names='col1, col2, col3',
                                     formats = 'S8, f8, i8')

#. Comsider a large vector Z, compute Z to the power of 3 using 3 different
   methods (★★★)
   
   .. code-block:: python

      Author: Ryan G.
                   
      x = np.random.rand(5e7)

      %timeit np.power(x,3)
      1 loops, best of 3: 574 ms per loop

      %timeit x*x*x
      1 loops, best of 3: 429 ms per loop

      %timeit np.einsum('i,i,i->i',x,x,x)
      1 loops, best of 3: 244 ms per loop
 
                   
#. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A
   that contain elements of each row of B regardless of the order of the elements
   in B ? (★★★) 

   .. code-block:: python

      # Author: Gabe Schwartz

      A = np.random.randint(0,5,(8,3))
      B = np.random.randint(0,5,(2,2))

      C = (A[..., np.newaxis, np.newaxis] == B)
      rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
      print(rows)


#. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★) 

   .. code-block:: python

      # Author: Robert Kern

      Z = np.random.randint(0,5,(10,3))
      E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
      U = Z[~E]
      print(Z)
      print(U)

      
#. Convert a vector of ints into a matrix binary representation (★★★) 

   .. code-block:: python

      # Author: Warren Weckesser

      I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
      B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
      print(B[:,::-1])

      # Author: Daniel T. McDonald

      I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
      print(np.unpackbits(I[:, np.newaxis], axis=1))


#. Given a two dimensional array, how to extract unique rows ? (★★★) 

   .. code-block:: python

      # Author: Jaime Fernández del Río

      Z = np.random.randint(0,2,(6,3))
      T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
      _, idx = np.unique(T, return_index=True)
      uZ = Z[idx]
      print(uZ)

#. Considering 2 vectors A & B, write the einsum equivalent of inner, outer,
   sum, and mul function (★★★) 


   .. code-block:: python

      # Author: Alex Riley
      # Make sure to read: http://ajcr.net/Basic-guide-to-einsum/

      np.einsum('i->', A)       # np.sum(A)
      np.einsum('i,i->i', A, B) # A * B
      np.einsum('i,i', A, B)	# np.inner(A, B)
      np.einsum('i,j', A, B)    # np.outer(A, B)


#. Considering a path described by two vectors (X,Y), how to sample it using
   equidistant samples (★★★) ?

   .. code-block:: python

      # Author: Bas Swinckels

      phi = np.arange(0, 10*np.pi, 0.1)
      a = 1
      x = a*phi*np.cos(phi)
      y = a*phi*np.sin(phi)

      dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
      r = np.zeros_like(x)
      r[1:] = np.cumsum(dr)                # integrate path
      r_int = np.linspace(0, r.max(), 200) # regular spaced path
      x_int = np.interp(r_int, r, x)       # integrate path
      y_int = np.interp(r_int, r, y)
