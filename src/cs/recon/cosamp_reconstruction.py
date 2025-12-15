# ===========================================================
# cosamp_reconstruction.py
# Implementação do algoritmo CoSaMP
# ===========================================================

import numpy as np

def cosamp_reconstruct(A, y, k, max_iter=10, tol=1e-6):
    """
    CoSaMP para resolver y ≈ A alpha com solução k-esparsa.

    Parâmetros:
    -----------
    A : ndarray (m x N)
        Matriz de medições (Phi @ D).
    y : ndarray (m,)
        Vetor de medições subamostradas.
    k : int
        Nível de sparsidade desejado.
    max_iter : int
        Número máximo de iterações.
    tol : float
        Tolerância para critério de parada.

    Retorna:
    --------
    alpha : ndarray (N,)
        Vetor de coeficientes esparsos estimado.
    """
    m, N = A.shape
    alpha = np.zeros(N)
    r = y.copy()

    for _ in range(max_iter):
        # 1) Correlação com o resíduo
        u = A.T @ r

        # 2) Seleciona 2k índices mais significativos
        idx_large = np.argsort(np.abs(u))[-2 * k:]

        # 3) Suporte candidato: suporte atual ∪ novos índices
        support_old = np.nonzero(alpha)[0]
        omega = np.union1d(support_old, idx_large)

        # 4) Resolve LS restrito a omega
        A_omega = A[:, omega]
        b_omega, *_ = np.linalg.lstsq(A_omega, y, rcond=None)

        alpha_temp = np.zeros(N)
        alpha_temp[omega] = b_omega

        # 5) Mantém apenas os k maiores coeficientes
        if k < N:
            support_new = np.argsort(np.abs(alpha_temp))[-k:]
        else:
            support_new = np.arange(N)

        alpha_new = np.zeros(N)
        alpha_new[support_new] = alpha_temp[support_new]

        # 6) Atualiza resíduo
        r = y - A @ alpha_new

        # 7) Critério de parada
        if np.linalg.norm(alpha_new - alpha) < tol * (np.linalg.norm(alpha_new) + 1e-12):
            alpha = alpha_new
            break

        alpha = alpha_new

    return alpha
