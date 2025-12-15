import numpy as np
from scipy.fftpack import dct

from src.analysis.thd_analysis import extract_harmonics


# ===========================================================
# Erros no tempo
# ===========================================================

def rmse(x, xhat):
    x = np.asarray(x).ravel()
    xhat = np.asarray(xhat).ravel()
    return np.sqrt(np.mean((x - xhat) ** 2))


def nrmse(x, xhat, eps=1e-12):
    """
    NRMSE = ||x - xhat||2 / ||x||2
    eps evita divisão por zero.
    """
    x = np.asarray(x).ravel()
    xhat = np.asarray(xhat).ravel()
    denom = np.linalg.norm(x)
    if denom < eps:
        return np.nan
    return np.linalg.norm(x - xhat) / denom


# ===========================================================
# Utilidades de CS / subamostragem
# ===========================================================

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


def reconstruct_signal_from_m(
    x, m, alg_name, RECON, D,
    indices=None, rng=None,
    k_rule="m_over_4",
    lambda_reg=5e-5
):
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

    xhat = D @ alpha
    return xhat


# ===========================================================
# Curva de erro no tempo por sinal
# ===========================================================

def rmse_curve_single_signal(
    x,
    RECON,
    max_m=64,
    min_m=3,
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices_per_m=True,
    use_nrmse=True,
):
    """
    Para 1 sinal x:
      retorna ms e dict {alg: [erro_tempo(m)]}
    """
    x = np.asarray(x).ravel()
    N = len(x)

    ms = list(range(max_m, min_m - 1, -1))
    out = {alg: [] for alg in RECON.keys()}

    D = dct(np.eye(N), norm="ortho")

    for m in ms:
        if m > N:
            for alg in out:
                out[alg].append(np.nan)
            continue

        rng = np.random.default_rng(fixed_seed + 1000 * m)

        # mesmos índices para todos os algoritmos (comparação justa)
        indices = np.sort(rng.choice(N, m, replace=False)) if fair_indices_per_m else None

        for alg in RECON.keys():
            xhat = reconstruct_signal_from_m(
                x, m, alg, RECON, D,
                indices=indices, rng=rng,
                k_rule=k_rule, lambda_reg=lambda_reg
            )
            err = nrmse(x, xhat) if use_nrmse else rmse(x, xhat)
            out[alg].append(err)

    return ms, out


# ===========================================================
# Espectro: erro agregado (vetor de harmônicas 1..H)
# ===========================================================

def harmonic_nrmse(
    original_signal,
    reconstructed_signal,
    fs,
    fundamental_freq,
    max_harmonic=31,
    window="hann",
    eps=1e-12
):
    """
    NRMSE no domínio da frequência, usando vetor de harmônicas 1..max_harmonic.

    NRMSE = ||H_rec - H_orig||2 / (||H_orig||2 + eps)
    """
    H_orig = extract_harmonics(
        original_signal,
        fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )
    H_rec = extract_harmonics(
        reconstructed_signal,
        fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )

    num = np.linalg.norm(H_rec - H_orig)
    den = np.linalg.norm(H_orig) + eps
    return float(num / den)


def spectral_nrmse_curve_single_signal(
    x,
    RECON,
    fs=3840,
    fundamental_freq=60,
    max_harmonic=31,
    window="hann",
    max_m=64,
    min_m=3,
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices_per_m=True,
):
    """
    Para 1 sinal x:
      - reconstrói xhat para cada m e algoritmo
      - calcula NRMSE espectral (harmônicas 1..max_harmonic) para cada m

    Retorna:
      ms, spec_curves  onde spec_curves = {alg: [nrmse_espectral(m)]}
    """
    x = np.asarray(x).ravel()
    N = len(x)

    ms = list(range(max_m, min_m - 1, -1))
    spec_curves = {alg: [] for alg in RECON.keys()}

    D = dct(np.eye(N), norm="ortho")

    for m in ms:
        if m > N:
            for alg in spec_curves:
                spec_curves[alg].append(np.nan)
            continue

        rng = np.random.default_rng(fixed_seed + 1000 * m)
        indices = np.sort(rng.choice(N, m, replace=False)) if fair_indices_per_m else None

        for alg in RECON.keys():
            xhat = reconstruct_signal_from_m(
                x, m, alg, RECON, D,
                indices=indices, rng=rng,
                k_rule=k_rule, lambda_reg=lambda_reg
            )

            spec_err = harmonic_nrmse(
                original_signal=x,
                reconstructed_signal=xhat,
                fs=fs,
                fundamental_freq=fundamental_freq,
                max_harmonic=max_harmonic,
                window=window,
            )

            spec_curves[alg].append(spec_err if np.isfinite(spec_err) else np.nan)

    return ms, spec_curves


# ===========================================================
# NOVO: tabela por harmônica (Opção A) para um m fixo
# ===========================================================

def _harmonic_nrmse_per_h(H_orig, H_rec, eps=1e-12):
    """
    NRMSE por harmônica:
      err[h] = |H_rec[h] - H_orig[h]| / (|H_orig[h]| + eps)
    """
    H_orig = np.asarray(H_orig, dtype=float)
    H_rec = np.asarray(H_rec, dtype=float)
    return np.abs(H_rec - H_orig) / (np.abs(H_orig) + eps)


def harmonic_error_table_for_m(
    signals,
    RECON,
    m_ref,
    fs=3840,
    fundamental_freq=60.0,
    max_harmonic=31,
    window="hann",
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices=True,
    eps=1e-12,
):
    """
    Gera tabela (por harmônica 1..max_harmonic) com:
      média e std do erro por harmônica para cada algoritmo, EM UM m FIXO.

    Retorna dict:
      {
        "harmonics": [1..H],
        "mean": {alg: [mean_err_h]},
        "std":  {alg: [std_err_h]},
        "n_signals_used": int,
      }
    """
    if len(signals) == 0:
        return {
            "harmonics": list(range(1, max_harmonic + 1)),
            "mean": {alg: [np.nan] * max_harmonic for alg in RECON.keys()},
            "std":  {alg: [np.nan] * max_harmonic for alg in RECON.keys()},
            "n_signals_used": 0,
        }

    # acumuladores por (alg, h)
    errs_by_alg = {alg: [[] for _ in range(max_harmonic)] for alg in RECON.keys()}
    used = 0

    for i, x in enumerate(signals):
        x = np.asarray(x).ravel()
        N = len(x)
        if m_ref > N:
            continue

        # base DCT
        D = dct(np.eye(N), norm="ortho")

        # índices fixos (mesmos para todos os algs) -> comparação justa no m_ref
        rng = np.random.default_rng(fixed_seed + 10000 * i + 1000 * m_ref)
        indices = np.sort(rng.choice(N, m_ref, replace=False)) if fair_indices else None

        # harmônicas do original (pré-computa)
        H0 = extract_harmonics(
            x,
            fs=fs,
            fundamental_freq=fundamental_freq,
            max_harmonic=max_harmonic,
            window=window
        )

        ok_this = False

        for alg in RECON.keys():
            xhat = reconstruct_signal_from_m(
                x, m_ref, alg, RECON, D,
                indices=indices,
                rng=rng,
                k_rule=k_rule,
                lambda_reg=lambda_reg
            )

            H1 = extract_harmonics(
                xhat,
                fs=fs,
                fundamental_freq=fundamental_freq,
                max_harmonic=max_harmonic,
                window=window
            )

            eh = _harmonic_nrmse_per_h(H0, H1, eps=eps)

            # guarda por harmônica
            for h_idx in range(max_harmonic):
                v = eh[h_idx]
                if np.isfinite(v):
                    errs_by_alg[alg][h_idx].append(float(v))

            ok_this = True

        if ok_this:
            used += 1

    # monta mean/std
    mean = {}
    std = {}
    for alg in RECON.keys():
        mean_vals = []
        std_vals = []
        for h_idx in range(max_harmonic):
            vals = np.asarray(errs_by_alg[alg][h_idx], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                mean_vals.append(np.nan)
                std_vals.append(np.nan)
            else:
                mean_vals.append(float(np.mean(vals)))
                std_vals.append(float(np.std(vals)))
        mean[alg] = mean_vals
        std[alg] = std_vals

    return {
        "harmonics": list(range(1, max_harmonic + 1)),
        "mean": mean,
        "std": std,
        "n_signals_used": used,
    }


# ===========================================================
# Média no tempo (vários sinais)
# ===========================================================

def avg_rmse_vs_m(
    signals,
    RECON,
    max_m=64,
    min_m=3,
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices_per_m=True,
    use_nrmse=True,
):
    """
    Curva média no TEMPO sobre vários sinais.
    """
    ms = list(range(max_m, min_m - 1, -1))
    if len(signals) == 0:
        return ms, {alg: [np.nan] * len(ms) for alg in RECON.keys()}

    sum_by_alg = {alg: np.zeros(len(ms), dtype=float) for alg in RECON.keys()}
    cnt_by_alg = {alg: np.zeros(len(ms), dtype=int) for alg in RECON.keys()}

    pos = {m: j for j, m in enumerate(ms)}

    for i, x in enumerate(signals):
        ms_i, curves = rmse_curve_single_signal(
            x,
            RECON,
            max_m=max_m,
            min_m=min_m,
            k_rule=k_rule,
            lambda_reg=lambda_reg,
            fixed_seed=fixed_seed + 10000 * i,
            fair_indices_per_m=fair_indices_per_m,
            use_nrmse=use_nrmse,
        )

        for alg, vals in curves.items():
            for m_val, val in zip(ms_i, vals):
                j = pos.get(m_val, None)
                if j is None or not np.isfinite(val):
                    continue
                sum_by_alg[alg][j] += float(val)
                cnt_by_alg[alg][j] += 1

    avg = {}
    for alg in RECON.keys():
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_vals = sum_by_alg[alg] / cnt_by_alg[alg]
        avg_vals[cnt_by_alg[alg] == 0] = np.nan
        avg[alg] = avg_vals.tolist()

    return ms, avg


# ===========================================================
# Média no espectro (vários sinais) — NRMSE agregado
# ===========================================================

def avg_spectral_nrmse_vs_m(
    signals,
    RECON,
    fs=3840,
    fundamental_freq=60,
    max_harmonic=31,
    window="hann",
    max_m=64,
    min_m=3,
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices_per_m=True,
):
    """
    Curva média no ESPECTRO (harmônicas 1..max_harmonic) sobre vários sinais.

    Retorna:
      ms, avg_spec  onde avg_spec = {alg: [nrmse_espectral_medio(m)]}
    """
    ms = list(range(max_m, min_m - 1, -1))
    if len(signals) == 0:
        return ms, {alg: [np.nan] * len(ms) for alg in RECON.keys()}

    sum_by_alg = {alg: np.zeros(len(ms), dtype=float) for alg in RECON.keys()}
    cnt_by_alg = {alg: np.zeros(len(ms), dtype=int) for alg in RECON.keys()}

    pos = {m: j for j, m in enumerate(ms)}

    for i, x in enumerate(signals):
        ms_i, spec_curves = spectral_nrmse_curve_single_signal(
            x,
            RECON,
            fs=fs,
            fundamental_freq=fundamental_freq,
            max_harmonic=max_harmonic,
            window=window,
            max_m=max_m,
            min_m=min_m,
            k_rule=k_rule,
            lambda_reg=lambda_reg,
            fixed_seed=fixed_seed + 10000 * i,
            fair_indices_per_m=fair_indices_per_m,
        )

        for alg, vals in spec_curves.items():
            for m_val, val in zip(ms_i, vals):
                j = pos.get(m_val, None)
                if j is None or not np.isfinite(val):
                    continue
                sum_by_alg[alg][j] += float(val)
                cnt_by_alg[alg][j] += 1

    avg_spec = {}
    for alg in RECON.keys():
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_vals = sum_by_alg[alg] / cnt_by_alg[alg]
        avg_vals[cnt_by_alg[alg] == 0] = np.nan
        avg_spec[alg] = avg_vals.tolist()

    return ms, avg_spec

def harmonic_error_table_for_m_long(
    signals,
    RECON,
    m_ref,
    fs=3840,
    fundamental_freq=60.0,
    max_harmonic=31,
    window="hann",
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices=True,
):
    """
    Retorna lista de dicts em formato 'long':
    [{m, harmonic, alg, mean, std, n}, ...]
    """
    # usa sua função existente harmonic_error_table_for_m(...)
    res = harmonic_error_table_for_m(
        signals=signals,
        RECON=RECON,
        m_ref=m_ref,
        fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window,
        k_rule=k_rule,
        lambda_reg=lambda_reg,
        fixed_seed=fixed_seed,
        fair_indices=fair_indices,
    )

    rows = []
    Hs = res["harmonics"]
    n_used = res["n_signals_used"]

    for i, h in enumerate(Hs):
        for alg in ("OMP", "CoSaMP", "BP"):
            rows.append({
                "m": int(m_ref),
                "harmonic": int(h),
                "alg": alg,
                "mean": float(res["mean"][alg][i]),
                "std": float(res["std"][alg][i]),
                "n": int(n_used),
            })
    return rows