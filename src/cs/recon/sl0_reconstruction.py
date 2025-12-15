# ===========================================================
# sl0_reconstruction.py
# Implementação simplificada do algoritmo SL0 (Smoothed-L0)
# Mohimani et al., 2008
# ===========================================================

import numpy as np

def sl0_reconstruct(A, y, k,
                    sigma_start=1.0,
                    sigma_end=1e-3,
                    L=6,
                    mu=2.0,
                    iters_per_sigma=5):
    """
    Reconstrução k-esparsa usando o algoritmo SL0 (Smoothed-L0).

    Resolve, de forma aproximada:
        min ||alpha||_0   sujeito a   y ≈ A alpha

    Parâmetros
    ----------
    A : ndarray (m x N)
        Matriz de medições (Phi @ D).
    y : ndarray (m,)
        Vetor de medições subamostradas.
    k : int
        Número desejado de coeficientes não nulos na solução final.
    sigma_start : float
        Sigma inicial (suavização mais "grossa").
    sigma_end : float
        Sigma final (suavização mais "fina").
    L : int
        Número de valores de sigma na sequência geométrica.
    mu : float
        Passo do gradiente (quanto maior, mais agressivo).
    iters_per_sigma : int
        Número de iterações de gradiente para cada sigma.

    Retorna
    -------
    alpha_sparse : ndarray (N,)
        Vetor de coeficientes esparsos estimado.
    """
    m, N = A.shape

    # --------------------------------------------
    # 1) Solução inicial: mínimos quadrados
    #    alpha0 = A^+ y
    # --------------------------------------------
    alpha = np.linalg.pinv(A) @ y

    At = A.T
    M = A @ At               # m x m
    # pré-calcula inversa para projeção (m pequeno → ok)
    M_inv = np.linalg.inv(M)

    # sequência de sigmas decrescentes (geométrica)
    sigmas = np.geomspace(sigma_start, sigma_end, L)

    for sigma in sigmas:
        for _ in range(iters_per_sigma):
            # ----------------------------------------
            # 2) "Gradiente" da função suavizada de L0
            #    phi(alpha) = sum(exp(-alpha_i^2 / (2 sigma^2)))
            #    queremos MAXIMIZAR phi → passo de subida
            # ----------------------------------------
            exp_term = np.exp(- (alpha**2) / (2.0 * sigma**2))
            grad = alpha * exp_term / (sigma**2)

            # passo de gradiente (subida)
            alpha = alpha + mu * grad

            # ----------------------------------------
            # 3) Projeção no conjunto de soluções de LS:
            #    proj(alpha) = argmin ||z - alpha||_2
            #    s.t. y = A z  (aproximado)
            #    = alpha + A^T (AA^T)^(-1) (y - A alpha)
            # ----------------------------------------
            r = y - A @ alpha               # resíduo de medição
            correction = At @ (M_inv @ r)   # A^T (AA^T)^(-1) r
            alpha = alpha + correction

    # --------------------------------------------
    # 4) Hard-threshold final: mantém só os k maiores
    # --------------------------------------------
    alpha_sparse = np.zeros_like(alpha)
    if k >= N:
        alpha_sparse = alpha
    else:
        idx = np.argsort(np.abs(alpha))[-k:]
        alpha_sparse[idx] = alpha[idx]

    return alpha_sparse
