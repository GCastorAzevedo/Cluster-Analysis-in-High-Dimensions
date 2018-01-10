# Cluster-Analysis-in-High-Dimensions
My master's thesis, accompanied with some (non-optimized) implementation of the Orclus algorithm (by Charu Aggarwal)

Here I present my master's thesis, together with two important papers that underlined the development of the main algorithm of this work. The thesis can be used as an introduction to Clustering Analysis, so many others references can be found on the bibliography.

The two articles uploaded in this repository can be accessed for free at: http://charuaggarwal.net/research.html . I highly recommend to acces other resources from this page, as its author, Charu Aggarwal is a prolific researcher in many Machine Learning domains of study.

I also give some code implementing the ORCLUS algorithm (depicted in "Finding Generalized Projected Clusters in High Dimensional Spaces"). This is just a sketch, absolutely non-optimized. My plans are to rewrite it on C, allowing some parallel computing enhancements, embedding afterwards into a newly optimized Pyhon package.

The package is delivered under the name "ClusterValidity.py", and it contains a family of cluster validity measures, again, absolutely non-optimized, designed specially for numpy arrays as inputs, then it contains some wrappers for scikit learn clustering procedures, and finally the ORCLUS "sketch". I also give an isolated commented implementation of the ORCLUS.

This is just an initial repository, that will be enhanced little by little.

For R developpers, excelent algorithms can be found at https://crantastic.org/packages/orclus .
