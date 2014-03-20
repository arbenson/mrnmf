MapReduce codes for large-scale nonnegative matrix factorization of near-separable
tall-and-skinny matrices.

This code needs the following software to run:
* Hadoop
* Dumbo + Feathers add-on
* NumPy + SciPy

The main Dumbo script is `RunNMF.py`.
Let X be the nonnegative data matrix.
This script supports computing any of the following:

* Gaussian projection: G * X, where G has Gaussian i.i.d. random entries
* TSQR: the R factor of X = QR
* Column l_1 norms: | X(:, i) |_1 for each column index i

These are all computed in one pass over the data.
See the documentation in the file for more information.
Here is an example command that does all three computatations:

     dumbo start RunNMF.py -mat data_matrix -output proj.out -blocksize 3 \
     -libjar feathers.jar -reduce_schedule 40,1 -hadoop icme-hadoop1

Code for plots that appear in the paper are in the plotting directory.
There, you can find the calls to SciPy's non-negative least squares solver.

Please contact Austin Benson (arbenson@stanford.edu) for help with running the code.
