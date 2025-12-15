import numpy as np
from scipy.fftpack import dct

def rmse(x, xhat):
    x = np.asarray(x).ravel()
    xhat = np.asarray(xhat).ravel()
    return np.sqrt(np.mean((x - xhat) ** 2))

def k_from_m(m, rule="m_over_4", k_max=31):
    if rule == "m_over_4":
        k = int(m // 4)
    elif rule == "m_over_2":
        k = int(m // 2)
    elif rule == "fixed10":
        k = 10
    else:
        k = 10
    return max(1, min(k, k_max, m - 1))

def build_measurement_matrix(N, m, D, indices=None, rng=None):
    if indices is None:
        rng = np.random.default_rng(0) if rng is None else rng
        indices = np.sort(rng.choice(N, m, replace=False))

    Phi = np.zeros((m, N))
    Phi[np.arange(m), indices] = 1.0
    A = Phi @ D
    return A, indices

def reconstruct_signal_from_m(x, m, alg_name, D, RECON,
                              indices=None, rng=None,
                              k_rule="m_over_4", lambda_reg=5e-5):
    x = np.asarray(x).ravel()
    N = len(x)
    if m > N:
        raise ValueError(f"m={m} maior que N={N}")

    A, indices = build_measurement_matrix(N, m, D, indices=indices, rng=rng)
    y = x[indices]

    if alg_name in ("OMP", "CoSaMP"):
        k = k_from_m(m, rule=k_rule, k_max=min(31, N))
        alpha = RECON[alg_name](A, y, k)
    elif alg_name == "BP":
        alpha = RECON[alg_name](A, y, lambda_reg=lambda_reg)
    else:
        raise ValueError(f"Algoritmo desconhecido: {alg_name}")

    return D @ alpha

def avg_rmse_vs_m(signals, RECON, max_m=64, min_m=3,
                  k_rule="m_over_4", lambda_reg=5e-5,
                  fixed_seed=0, fair_indices_per_m=True):
    ms = list(range(max_m, min_m - 1, -1))
    out = {alg: [] for alg in RECON.keys()}

    for m in ms:
        for alg in RECON.keys():
            errs = []
            for si, x in enumerate(signals):
                x = np.asarray(x).ravel()
                N = len(x)
                if N < m:
                    continue

                D = dct(np.eye(N), norm="ortho")
                rng = np.random.default_rng(fixed_seed + 1000*m + si)

                indices = np.sort(rng.choice(N, m, replace=False)) if fair_indices_per_m else None

                xhat = reconstruct_signal_from_m(
                    x, m, alg, D, RECON,
                    indices=indices, rng=rng,
                    k_rule=k_rule, lambda_reg=lambda_reg
                )
                errs.append(rmse(x, xhat))

            out[alg].append(np.mean(errs) if errs else np.nan)

    return ms, out
