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

This code is provided under the BSD 2-Clause License,
which can be found in the LICENSE file in the root directory, or at 
http://opensource.org/licenses/BSD-2-Clause.

If you use this code in a publication, please cite:

Benson, Austin R., Lee, Jason D., Rajwa, Bartek, and Gleich, David F.
"Scalable methods for nonnegative matrix factorizations of near-separable tall-and-skinny matrices."
arXiv preprint arXiv:1402.6964 (2014).


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

The main script for computing the extreme columns and the
coefficient matrix _H_ is `NMFProcessAlgorithms.py`.

Code for reproducing the plots in the paper are in the `plotting` directory.


Full small example
--------
A small noisy r-separable matrix is available in `SmallNoisySep_10k_10_4.txt`.
The matrix has 10,000 rows, 10 columns, and a separation rank of 4 (r = 4),
and was generated with `util_scripts/GenSyntheticSepSmall.py`.  The noise
level was 1e-3.
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

We see that the noise level for SPA is near 1e-3, as expected.
Since X-ray is greedy, it may not get all of the columns with the target
separation rank set to 4:

     python NMFProcessAlgorithms.py small.out-qrr.txt small.out-colnorms.txt 'xray' 5


Partial computation
--------
We do not need to compute the Gaussian projection, R factor, and column factors.
The script `RunNMF.py` supports running only a subset of these operations.

We will continue using the data from the small example above.
If we only wanted the R factor in the QR factorization:

     dumbo start RunNMF.py -libjar feathers.jar -hadoop $HADOOP_INSTALL \
     -reduce_schedule 2,1 -mat SmallNoisySep_10k_10_4.txt -output small.out \
     -gp 0 -colnorms 0

The last two options specify to _not_ do Gaussian projection or compute the
column norms.
Now, we see that only the R factor has been computed:

     hadoop fs -ls small.out

Alternatively, we could just omit the QR computation:

     dumbo start RunNMF.py -libjar feathers.jar -hadoop $HADOOP_INSTALL \
     -projsize 10 -reduce_schedule 2,1 -mat SmallNoisySep_10k_10_4.txt -output small.out \
     -qr 0

The computation of the R factor has been omitted:

     hadoop fs -ls small.out


Flow cytometry (FC) setup
--------

The original FC measurements are in `data/FC_40k.txt`.
We are interested in the Kronecker product of this matrix.
We form the Kronecker product with MapReduce.
First, copy the data to HDFS:

     hadoop fs -put data/FC_40k.txt FC_40k.txt

Now, run the MapReduce job to form the Kronecker product:

     dumbo start FC_kron.py -mat FC_40k.txt -output FC_kron.bseq -hadoop $HADOOP_INSTALL

The rows of the matrix are grouped with 1000 rows per record.
With the Kronecker product on HDFS, we can do dimension reduction:

     dumbo start RunNMF.py -libjar feathers.jar -hadoop $HADOOP_INSTALL \
     -reduce_schedule 40,1 -mat FC_kron.bseq -output FC_data.out

Large synthetic matrices setup
--------

The synthetic coefficient matrix used in the paper is in `data/Hprime_20_200.txt`.
We can generate (noisy) r-separable matrices using the script `GenSyntheticSepLarge.py`.
To generate a 200M x 200 matrix with r = 20 and noise level epsilon=1e-3:

     hadoop fs -put data/Simple_1M.txt Simple_1M.txt     

     dumbo start GenSyntheticSepLarge.py -hadoop $HADOOP_INSTALL \
     -m 200 -H 'data/Hprime_20_200.txt' -epsilone 1e-3 \
     -mat Simple_1M.txt -output Noisy_200M_200_20.bseq

Contact
--------
Please contact Austin Benson (arbenson@stanford.edu) for help with the code, bug reports, and feature requests.
