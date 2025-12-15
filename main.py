# ===========================================================
# main.py
# Script principal do artigo de reconstrução de sinais
# ===========================================================

import sys
from pathlib import Path

# ===========================================================
# Ajuste de PATH (raiz do projeto)
# ===========================================================
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import numpy as np

# ===========================================================
# Imports do projeto
# ===========================================================
from src.io.io_signals import list_locais, collect_signals
from src.analysis.thd_analysis import (run_thd_analysis,extract_harmonics, harmonic_nrmse)
from src.analysis.metrics import (avg_rmse_vs_m,harmonic_error_table_for_m_long)  # agora pode retornar NRMSE se use_nrmse=True

from src.cs.recon.omp_reconstruction import omp_reconstruct
from src.cs.recon.cosamp_reconstruction import cosamp_reconstruct
from src.cs.recon.bp_reconstruction import bp_reconstruct

from src.analysis.per_signal_report import report_thd_and_rmse_per_signal
from src.plots.plots import plot_avg_curves

from src.plots.plot_harmonic_table_csv import plot_harmonics_from_csv

from src.plots.plot_correlation import (
    plot_correlation_spearman_vs_m,
    plot_correlation_pearson_vs_m,
    plot_correlation_kendall_vs_m,
)

import csv
import pandas as pd
# ===========================================================
# Configurações globais
# ===========================================================
MAX_M = 64
MIN_M = 3
K_RULE = "m_over_4"
LAMBDA_BP = 5e-5
MAX_HARMONIC = 31  # usado no THD

RECON = {
    "OMP": omp_reconstruct,
    "CoSaMP": cosamp_reconstruct,
    "BP": bp_reconstruct,
}

# ===========================================================
# MAIN
# ===========================================================
def main():

    # -------------------------------------------------------
    # 1) Descobrir locais
    # -------------------------------------------------------
    locais = list_locais("dados")
    print("\nLocais detectados:", locais)

    # -------------------------------------------------------
    # 2) Carregar sinais
    # -------------------------------------------------------
    tensoes, tensao_paths = collect_signals(locais, "tensao")
    correntes, corrente_paths = collect_signals(locais, "corrente")

    print(f"\nTotal tensões:    {len(tensoes)}")
    print(f"Total correntes:  {len(correntes)}")

    if not tensoes and not correntes:
        raise RuntimeError("Nenhum sinal encontrado.")

    # -------------------------------------------------------
    # 3) THD
    # -------------------------------------------------------
    thd_t = run_thd_analysis(tensoes, tensao_paths, max_harmonic=MAX_HARMONIC, label="tensões")
    thd_c = run_thd_analysis(correntes, corrente_paths, max_harmonic=MAX_HARMONIC, label="correntes")

    thd_all = thd_t + thd_c
    if thd_all:
        print(f"\nTHD médio geral: {np.mean(thd_all):.2f} %")

    # -------------------------------------------------------
    # 4) ERRO médio vs m (NRMSE recomendado)
    # -------------------------------------------------------
    print("\n=== Calculando NRMSE médio vs m ===")

    ms_t, err_t = avg_rmse_vs_m(
        tensoes,
        RECON,
        max_m=MAX_M,
        min_m=MIN_M,
        k_rule=K_RULE,
        lambda_reg=LAMBDA_BP,
        fixed_seed=0,
        fair_indices_per_m=True,
        use_nrmse=True,   # ✅ AQUI
    )

    ms_c, err_c = avg_rmse_vs_m(
        correntes,
        RECON,
        max_m=MAX_M,
        min_m=MIN_M,
        k_rule=K_RULE,
        lambda_reg=LAMBDA_BP,
        fixed_seed=123,
        fair_indices_per_m=True,
        use_nrmse=True,   # ✅ AQUI
    )

    ms_g, err_g = avg_rmse_vs_m(
        tensoes + correntes,
        RECON,
        max_m=MAX_M,
        min_m=MIN_M,
        k_rule=K_RULE,
        lambda_reg=LAMBDA_BP,
        fixed_seed=999,
        fair_indices_per_m=True,
        use_nrmse=True,   # ✅ AQUI
    )

    # -------------------------------------------------------
    # 4.1) IMPRIMIR NO CONSOLE — NRMSE médio vs m (GERAL)
    # -------------------------------------------------------
    print("\n=== NRMSE MÉDIO vs m (GERAL: tensões + correntes) ===")
    print(" m |      OMP         CoSaMP          BP")
    print("-" * 55)

    for i, m in enumerate(ms_g):
        omp = err_g["OMP"][i]
        cos = err_g["CoSaMP"][i]
        bp  = err_g["BP"][i]
        print(f"{m:3d} | {omp:12.4e}  {cos:12.4e}  {bp:12.4e}")

    # -------------------------------------------------------
    # 4.2) RELATÓRIO SINAL-A-SINAL (THD + curva de erro)
    # -------------------------------------------------------
    print("\n### RELATÓRIO SINAL-A-SINAL: TENSÕES ###")
    rows_t = report_thd_and_rmse_per_signal(
        tensoes, tensao_paths, RECON,
        max_m=MAX_M, min_m=MIN_M,
        k_rule=K_RULE,
        lambda_reg=LAMBDA_BP,
        print_full_curve=False,
        low_m_threshold=10
    )

    print("\n### RELATÓRIO SINAL-A-SINAL: CORRENTES ###")
    rows_c = report_thd_and_rmse_per_signal(
        correntes, corrente_paths, RECON,
        max_m=MAX_M, min_m=MIN_M,
        k_rule=K_RULE,
        lambda_reg=LAMBDA_BP,
        print_full_curve=False,
        low_m_threshold=10
    )

    # -------------------------------------------------------
    # 4.3) CORRELAÇÃO: THD × ERRO(m) — plots (Spearman/Pearson/Kendall)
    # -------------------------------------------------------

    # -------------------------------------------------------
# 4.3) CORRELAÇÃO: THD × ERRO(m) — comparação TENSÃO × CORRENTE
# -------------------------------------------------------

    # =========================
    # APENAS TENSÕES
    # =========================
    print("\n### CORRELAÇÃO: THD × ERRO(m FIXO) — APENAS TENSÕES ###")

    plot_correlation_spearman_vs_m(
        rows_t,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Tensões — Correlação THD × Erro(m) (Spearman ρ)",
    )

    plot_correlation_pearson_vs_m(
        rows_t,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Tensões — Correlação THD × Erro(m) (Pearson r)",
    )

    plot_correlation_kendall_vs_m(
        rows_t,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Tensões — Correlação THD × Erro(m) (Kendall τ)",
    )


    # =========================
    # APENAS CORRENTES
    # =========================
    print("\n### CORRELAÇÃO: THD × ERRO(m FIXO) — APENAS CORRENTES ###")

    plot_correlation_spearman_vs_m(
        rows_c,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Correntes — Correlação THD × Erro(m) (Spearman ρ)",
    )

    plot_correlation_pearson_vs_m(
        rows_c,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Correntes — Correlação THD × Erro(m) (Pearson r)",
    )

    plot_correlation_kendall_vs_m(
        rows_c,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Correntes — Correlação THD × Erro(m) (Kendall τ)",
    )


    # =========================
    # GERAL (TENSÕES + CORRENTES)
    # =========================
    print("\n### CORRELAÇÃO: THD × ERRO(m FIXO) — GERAL ###")

    rows_all = rows_t + rows_c

    plot_correlation_spearman_vs_m(
        rows_all,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Geral — Correlação THD × Erro(m) (Spearman ρ)",
    )

    plot_correlation_pearson_vs_m(
        rows_all,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Geral — Correlação THD × Erro(m) (Pearson r)",
    )

    plot_correlation_kendall_vs_m(
        rows_all,
        algorithms=("OMP", "CoSaMP", "BP"),
        title="Geral — Correlação THD × Erro(m) (Kendall τ)",
    )


    # -------------------------------------------------------
    # 5) Gráficos (opcional)
    # -------------------------------------------------------
    plot_avg_curves(ms_t, err_t, "NRMSE médio vs m — TENSÕES")
    plot_avg_curves(ms_c, err_c, "NRMSE médio vs m — CORRENTES")
    plot_avg_curves(ms_g, err_g, "NRMSE médio vs m — GERAL")



# -------------------------------------------------------
# 6) ANÁLISE ESPECTRAL — TABELAS POR HARMÔNICA (Opção A)
# -------------------------------------------------------
    print("\n=== ITEM 6: TABELAS ESPECTRAIS (erro por harmônica) ===")

    from src.analysis.metrics import harmonic_error_table_for_m

    FS = 3840          # ajuste se necessário
    F0 = 60.0
    MAX_H = 31
    WINDOW = "hann"

    M_LIST = list(range(64, 2, -1))# suas 7 escolhas

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    def export_group(signals, group_name, seed_base):
        out_path = out_dir / f"harmonic_errors_{group_name.lower()}.csv"

        all_rows = []
        for m_ref in M_LIST:
            rows = harmonic_error_table_for_m_long(
                signals=signals,
                RECON=RECON,
                m_ref=m_ref,
                fs=FS,
                fundamental_freq=F0,
                max_harmonic=MAX_H,
                window=WINDOW,
                k_rule=K_RULE,
                lambda_reg=LAMBDA_BP,
                fixed_seed=seed_base + 100000*m_ref,
                fair_indices=True,
            )
            for r in rows:
                r["group"] = group_name
            all_rows.extend(rows)

        # escreve CSV
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["group","m","harmonic","alg","mean","std","n"])
            w.writeheader()
            w.writerows(all_rows)

        print(f"[OK] CSV salvo em: {out_path}")

    export_group(tensoes, "TENSOES", seed_base=0)
    export_group(correntes, "CORRENTES", seed_base=123)


    plot_harmonics_from_csv("./outputs/harmonic_errors_correntes.csv", group="CORRENTES", harmonics=(3,5,7,9,11), alg="OMP" , show_std=False)
    plot_harmonics_from_csv("./outputs/harmonic_errors_correntes.csv", group="CORRENTES", harmonics=(3,5,7,9,11), alg="BP", show_std=False)
    plot_harmonics_from_csv("./outputs/harmonic_errors_correntes.csv", group="CORRENTES", harmonics=(3,5,7,9,11), alg="CoSaMP", show_std=False)
    plot_harmonics_from_csv("./outputs/harmonic_errors_tensoes.csv", group="TENSOES", harmonics=(3,5,7,9,11), alg="OMP", show_std=False)
    plot_harmonics_from_csv("./outputs/harmonic_errors_tensoes.csv", group="TENSOES", harmonics=(3,5,7,9,11), alg="BP", show_std=False)
    plot_harmonics_from_csv("./outputs/harmonic_errors_tensoes.csv", group="TENSOES", harmonics=(3,5,7,9,11), alg="CoSaMP", show_std=False)

    print("\n=== Execução finalizada com sucesso ===")


# ===========================================================
if __name__ == "__main__":
    main()
