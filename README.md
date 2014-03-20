MapReduce Nonnegative Matrix Factorizations of near-separable tall-and-skinny matrices.
--------
Austin R. Benson, Jason D. Lee, Bartek Rajwa, and David F. Gleich.

This code provides a MapReduce implementation of the large-scale near-separable nonnegative matrix
factorizations described in
[Scalable methods for nonnegative matrix factorizations of near-separable tall-and-skinny matrices](http://arxiv.org/abs/1402.6964).
The implementation uses Hadoop with Python streaming, supported by Dumbo and Feathers.

Given a data matrix _X_ of size _m_ x _n_, with _m_ >> _n_ and nonnegative entries,
we are interested in a separable nonnegative matrix factorization:

     X ~ X(:, K)H,

where _X(:, K)_ is some permuted column subset of _X_ with _|K|_ columns,
and _H_ is _|K|_ x _n_ with nonnegative entries.

Setup
--------
This code needs the following software to run:
* Hadoop
* NumPy + SciPy
* [Dumbo](https://github.com/klbostee/dumbo/) + [Feathers](https://github.com/klbostee/feathers)

After building Feathers, place the jar file in the head of the `mrnmf` directory.
This will let you run the examples in this documentation.

We assume that the matrix data records in HDFS are one of the following:
* a tab or space-separated string representing one row
* a list representing one row
* a list of lists representing multiple rows
* a numpy.ndarray type representing multiple rows

Overview
--------
The main Dumbo script is `RunNMF.py`.
This is used to reduce the dimension of the problem.
The main script for computing the columns and the
coefficient matrix H is `NMFProcessAlgorithms.py`.
This script supports computing any of the following:

* Gaussian projection: G * X, where _G_ has Gaussian i.i.d. random entries
* TSQR: the R factor of X = QR
* Column l_1 norms: | X(:, i) |_1 for each column index i

These are all computed in one pass over the data.
See the documentation in the file for more information.
Here is an example command that does all three computatations:

     dumbo start RunNMF.py -mat datamatrix -output proj.out -projsize 100 \
     -libjar feathers.jar -reduce_schedule 40,1 -hadoop $HADOOP_INSTALL

The parameters are:
* datamatrix is the name of the matrix file in HDFS.
* projsize is the number of rows in G
* 40,1 means 40 reducers in the first pass and 1 reducer in the second pass
* $HADOOP_INSTALL is the location of Hadoop.

Code for plots that appear in the paper are in the plotting directory.
There, you can find the calls to SciPy's non-negative least squares solver.

Full small example
--------
A small noisy r-separable matrix is available in `SmallNoisySep_10k_10_4.txt`.
The matrix has 10,000 rows, 10 columns, and a separation rank of 4 (r = 4),
and was generated with `util_scripts/GenSyntheticSepSmall.py`.
First, put the matrix in HDFS:

     hadoop fs -put data/SmallNoisySep_10k_10_4.txt SmallNoisySep_10k_10_4.txt

Next, compute the Gaussian projection, R, and column norms:

     dumbo start RunNMF.py -libjar feathers.jar -hadoop $HADOOP_INSTALL \
     -projsize 12 -reduce_schedule 2,1 -mat SmallNoisySep_10k_10_4.txt -output small.out

Copy the data locally and look at the output.

     dumbo cat small.out/GP -hadoop $HADOOP_INSTALL > small.out-proj.txt
     cat small.out-proj.txt
     dumbo cat small.out/QR -hadoop $HADOOP_INSTALL > small.out-qrr.txt
     cat small.out-qrr.txt
     dumbo cat small.out/colnorms -hadoop $HADOOP_INSTALL > small.out-colnorms.txt
     cat small.out-colnorms.txt

Compute NMF with different algorithms:

     python NMFProcessAlgorithms.py small.out-qrr.txt small.out-colnorms.txt 'SPA' 4
     python NMFProcessAlgorithms.py small.out-qrr.txt small.out-colnorms.txt 'xray' 4
     python NMFProcessAlgorithms.py small.out-proj.txt small.out-colnorms.txt 'GP' 4

Since X-ray is greedy, it may not get all of the columns with the target
separation rank set to 4:

     python NMFProcessAlgorithms.py small.out-qrr.txt small.out-colnorms.txt 'xray' 5
     

Contact
--------
Please contact Austin Benson (arbenson@stanford.edu) for help with the code.
