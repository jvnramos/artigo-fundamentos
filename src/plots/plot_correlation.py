import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau


def plot_correlation_spearman_vs_m(
    rows,
    algorithms=("OMP", "CoSaMP", "BP"),
    title=None,          # 👈 NOVO
    show=True,
    save_path=None,
):
    """Correlação Spearman (ρ)"""
    return plot_correlation_vs_m(
        rows,
        algorithms=algorithms,
        method="spearman",
        title=title,
        show=show,
        save_path=save_path,
    )


def plot_correlation_pearson_vs_m(
    rows,
    algorithms=("OMP", "CoSaMP", "BP"),
    title=None,          # 👈 NOVO
    show=True,
    save_path=None,
):
    """Correlação Pearson (r)"""
    return plot_correlation_vs_m(
        rows,
        algorithms=algorithms,
        method="pearson",
        title=title,
        show=show,
        save_path=save_path,
    )


def plot_correlation_kendall_vs_m(
    rows,
    algorithms=("OMP", "CoSaMP", "BP"),
    title=None,          # 👈 NOVO
    show=True,
    save_path=None,
):
    """Correlação Kendall (τ)"""
    return plot_correlation_vs_m(
        rows,
        algorithms=algorithms,
        method="kendall",
        title=title,
        show=show,
        save_path=save_path,
    )


def plot_correlation_vs_m(
    rows,
    algorithms=("OMP", "CoSaMP", "BP"),
    method="spearman",   # spearman | pearson | kendall
    title=None,          # 👈 NOVO
    show=True,
    save_path=None,
):
    """
    Plota correlação(THD, ERRO[m]) vs m em UM ÚNICO gráfico,
    com curvas para OMP, CoSaMP e BP.
    """

    if not rows:
        print("[plot_correlation_vs_m] rows vazio; nada a plotar.")
        return

    # malha de m de referência
    ref_ms = None
    for r in rows:
        ms = r.get("ms", None)
        if ms:
            ref_ms = list(ms)
            break

    if ref_ms is None:
        print("[plot_correlation_vs_m] Não achei 'ms' em rows.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for alg in algorithms:
        corrs = []

        for m in ref_ms:
            xs, ys = [], []

            for r in rows:
                thd = r.get("thd", None)
                ms = r.get("ms", None)
                curves = r.get("rmse_curves", {})

                if thd is None or ms is None:
                    continue
                if not np.isfinite(thd):
                    continue
                if alg not in curves:
                    continue

                try:
                    j = ms.index(m)
                except ValueError:
                    continue

                y = curves[alg][j]
                if y is None or not np.isfinite(y):
                    continue

                xs.append(float(thd))
                ys.append(float(y))

            if len(xs) >= 3:
                if method == "spearman":
                    c, _ = spearmanr(xs, ys)
                elif method == "pearson":
                    c, _ = pearsonr(xs, ys)
                elif method == "kendall":
                    c, _ = kendalltau(xs, ys)
                else:
                    raise ValueError("method deve ser: spearman | pearson | kendall")

                corrs.append(float(c))
            else:
                corrs.append(np.nan)

        ax.plot(ref_ms, corrs, label=alg)

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("m")
    ax.set_ylabel("Correlação THD × Erro(m)")
    ax.set_ylim(-1, 1)
    ax.invert_xaxis()
    ax.grid(True)

    title_map = {
        "spearman": "Spearman (ρ)",
        "pearson": "Pearson (r)",
        "kendall": "Kendall (τ)",
    }

    ax.set_title(
        title if title is not None
        else f"Correlação vs m — {title_map[method]}"
    )

    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
