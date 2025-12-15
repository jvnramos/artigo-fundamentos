import numpy as np

from src.analysis.metrics import rmse_curve_single_signal
from src.analysis.thd_analysis import compute_thd


def nrmse_from_rmse(x, rmse_val, eps=1e-12):
    """
    Converte RMSE -> NRMSE usando normalização por ||x||_2 / sqrt(N).
    NRMSE = RMSE / RMS(x) , onde RMS(x)=||x||_2/sqrt(N)
    (equivalente a ||x-xhat||_2 / ||x||_2)
    """
    x = np.asarray(x).ravel()
    N = len(x)
    rms_x = (np.linalg.norm(x) / np.sqrt(max(N, 1)))  # RMS do sinal
    denom = max(rms_x, eps)
    return float(rmse_val) / denom


def summarize_curve(ms, nrmse_vals, low_m_threshold=10):
    """
    Extrai escalares da curva NRMSE(m) de um algoritmo:
      - nrmse_mean: média em todos os m
      - nrmse_low : média para m <= low_m_threshold
      - nrmse_at_minm: NRMSE no menor m
      - nrmse_at_maxm: NRMSE no maior m
      - slope_linear: inclinação aproximada via ajuste linear NRMSE ~ a*m + b
    """
    ms = np.asarray(ms, dtype=float)
    y = np.asarray(nrmse_vals, dtype=float)

    mask = np.isfinite(y)
    ms2 = ms[mask]
    y2 = y[mask]

    if len(y2) == 0:
        return {
            "nrmse_mean": np.nan,
            "nrmse_low": np.nan,
            "nrmse_at_minm": np.nan,
            "nrmse_at_maxm": np.nan,
            "slope_linear": np.nan,
        }

    nrmse_mean = float(np.mean(y2))

    mask_low = (ms2 <= low_m_threshold)
    nrmse_low = float(np.mean(y2[mask_low])) if np.any(mask_low) else np.nan

    nrmse_at_maxm = float(y2[0])   # ms começa no max_m
    nrmse_at_minm = float(y2[-1])  # termina no min_m

    if len(y2) >= 2:
        a, _b = np.polyfit(ms2, y2, 1)
        slope = float(a)
    else:
        slope = np.nan

    return {
        "nrmse_mean": nrmse_mean,
        "nrmse_low": nrmse_low,
        "nrmse_at_minm": nrmse_at_minm,
        "nrmse_at_maxm": nrmse_at_maxm,
        "slope_linear": slope,
    }


def report_thd_and_rmse_per_signal(
    signals,
    paths,
    RECON,
    max_m=64,
    min_m=3,
    k_rule="m_over_4",
    lambda_reg=5e-5,
    fixed_seed=0,
    fair_indices_per_m=True,
    print_full_curve=False,
    low_m_threshold=10
):
    """
    Para cada (x, path):
      - calcula THD(x)
      - calcula RMSE(m) para cada algoritmo (via rmse_curve_single_signal)
      - converte RMSE(m) -> NRMSE(m)
      - salva em rows (mantendo chave 'rmse_curves' por compatibilidade)
    """
    rows = []

    for i, (x, path) in enumerate(zip(signals, paths)):
        x = np.asarray(x).ravel()

        thd = compute_thd(x, max_harmonic=20)

        ms, rmse_curves = rmse_curve_single_signal(
            x,
            RECON,
            max_m=max_m,
            min_m=min_m,
            k_rule=k_rule,
            lambda_reg=lambda_reg,
            fixed_seed=fixed_seed + 10000 * i,
            fair_indices_per_m=fair_indices_per_m,
        )

        # Converte curvas RMSE -> NRMSE (por sinal)
        nrmse_curves = {}
        for alg, rmse_vals in rmse_curves.items():
            nrmse_vals = []
            for rmse_val in rmse_vals:
                if rmse_val is None or not np.isfinite(rmse_val):
                    nrmse_vals.append(np.nan)
                else:
                    nrmse_vals.append(nrmse_from_rmse(x, rmse_val))
            nrmse_curves[alg] = nrmse_vals

        # resumo por algoritmo (agora NRMSE)
        per_alg_summary = {}
        for alg, nrmse_vals in nrmse_curves.items():
            per_alg_summary[alg] = summarize_curve(ms, nrmse_vals, low_m_threshold=low_m_threshold)

        if print_full_curve:
            print("\nNRMSE(m) completo:")
            header = "m".rjust(4) + "  " + "  ".join([alg.rjust(12) for alg in RECON.keys()])
            print(header)
            print("-" * len(header))
            for j, m in enumerate(ms):
                line = str(m).rjust(4)
                for alg in RECON.keys():
                    val = nrmse_curves[alg][j]
                    line += f"  {val:12.4e}"
                print(line)

        rows.append({
            "path": path,
            "thd": thd,
            "ms": ms,

            # ✅ Mantém o nome antigo para não quebrar plot_correlation etc.
            # (mas agora contém NRMSE!)
            "rmse_curves": nrmse_curves,

            "summary": per_alg_summary,
        })

    return rows
