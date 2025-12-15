import numpy as np


# ===========================================================
# Correlações básicas
# ===========================================================
def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, 0

    r = np.corrcoef(x, y)[0, 1]
    return float(r), int(len(x))


def _spearman(x, y):
    """
    Spearman simples:
    correlação de Pearson aplicada aos ranks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, 0

    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)

    r = np.corrcoef(rx, ry)[0, 1]
    return float(r), int(len(x))


# ===========================================================
# Opção A — Correlação THD × RMSE(m) para m fixo
# ===========================================================
def correlations_thd_vs_rmse_per_m(
    rows,
    algorithms=("OMP", "CoSaMP", "BP"),
    use_spearman=True
):
    """
    Para cada algoritmo e para cada valor de m:
      calcula corr(THD, RMSE(m))

    rows: retorno de report_thd_and_rmse_per_signal(...)
    """

    # -------------------------------------------------------
    # Coleta THD e RMSE(m) por algoritmo
    # -------------------------------------------------------
    thd_vals = []
    rmse_by_alg = {alg: {} for alg in algorithms}

    for r in rows:
        thd = r.get("thd", np.nan)
        if not np.isfinite(thd):
            continue

        thd_vals.append(thd)
        ms = r["ms"]

        for alg in algorithms:
            curve = r["rmse_curves"][alg]
            for m, rmse in zip(ms, curve):
                rmse_by_alg[alg].setdefault(m, []).append(rmse)

    thd_vals = np.asarray(thd_vals, dtype=float)

    # -------------------------------------------------------
    # Impressão dos resultados
    # -------------------------------------------------------
    print("\n=== CORRELAÇÃO THD × RMSE(m FIXO) ===")
    corr_name = "Spearman ρ" if use_spearman else "Pearson r"
    print(f"Métrica de correlação: {corr_name}")
    print("-" * 60)

    results = {}  # útil se quiser salvar depois

    for alg in algorithms:
        print(f"\n>>> Algoritmo: {alg}")
        print(" m | correlação (n)")
        print("-" * 28)

        results[alg] = {}

        for m in sorted(rmse_by_alg[alg].keys(), reverse=True):
            rmse_vals = np.asarray(rmse_by_alg[alg][m], dtype=float)

            if use_spearman:
                r, n = _spearman(thd_vals, rmse_vals)
            else:
                r, n = _pearson(thd_vals, rmse_vals)

            results[alg][m] = r
            print(f"{m:3d} | {r:10.4f} ({n:2d})")

    print("\nInterpretação:")
    print(" r/ρ > 0  → THD maior tende a RMSE maior naquele m")
    print(" r/ρ ≈ 0  → THD pouco relevante naquele regime")
    print(" r/ρ < 0  → THD maior tende a RMSE menor (raro aqui)")

    return results
