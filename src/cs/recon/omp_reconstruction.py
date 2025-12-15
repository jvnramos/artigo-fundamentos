import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

def omp_reconstruct(A, y, k):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    omp.fit(A, y)
    return omp.coef_
