# ===========================================================
# bp_reconstruction.py
# Basis Pursuit (L1 minimization) usando LASSO
# ===========================================================

import numpy as np
from sklearn.linear_model import Lasso

def bp_reconstruct(A, y, lambda_reg=1e-3, max_iter=10000):
    """
    Reconstrução via Basis Pursuit (L1) relaxado:
        min ||y - A alpha||^2 + λ ||alpha||_1

    Parâmetros
    ----------
    A : ndarray (m x N)
        Matriz de medições subamostradas.
    y : ndarray (m,)
        Vetor de medições subamostradas.
    lambda_reg : float
        Parâmetro λ de regularização L1. 
        Quanto menor → mais fiel aos dados.
        Quanto maior → mais esparso, porém menos preciso.
    max_iter : int
        Número de iterações permitidas no solver.

    Retorna
    -------
    alpha : ndarray (N,)
        Coeficientes estimados (não necessariamente k-esparsos).
    """
    # Configura modelo LASSO para aproximar BP (Basis Pursuit relaxado)
    lasso = Lasso(alpha=lambda_reg,
                  fit_intercept=False,
                  max_iter=max_iter)

    # Lasso espera A e y como float64
    lasso.fit(A, y)

    return lasso.coef_
