# src/viz/recon_plots.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from src.cs.recon.omp_reconstruction import omp_reconstruct
from src.cs.recon.cosamp_reconstruction import cosamp_reconstruct
from src.cs.recon.bp_reconstruction import bp_reconstruct


def rmse(x, xhat):
    x = np.asarray(x).ravel()
    xhat = np.asarray(xhat).ravel()
    return np.sqrt(np.mean((x - xhat) ** 2))


def compute_fft(signal, fs):
    signal = np.asarray(signal).ravel()
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    mag = np.abs(fft_vals)
    return freqs, mag


def plot_reconstruction_matrix(
    x,
    m_values,
    k=10,
    lambda_reg=5e-5,
    seed=0,
    fs=3840,          # 64 amostras por ciclo de 60 Hz -> fs=3840
    max_harm=31       # para tabela espectral (ímpares até 31)
):
    """
    Matriz com:
      Coluna 1: sinal subamostrado
      Coluna 2: original + OMP + CoSaMP + BP
    + imprime tabelas: RMSE no tempo e comparação espectral (harmônicas ímpares).
    """
    x = np.asarray(x).ravel()
    N = len(x)
    n = np.arange(N)

    D = dct(np.eye(N), norm="ortho")
    rng = np.random.default_rng(seed)

    fig, axs = plt.subplots(len(m_values), 2, figsize=(14, 12))
    fig.suptitle("Subamostragem e Reconstrução – OMP, CoSaMP e BP (L1)", fontsize=16)

    rmse_omp_list, rmse_cosamp_list, rmse_bp_list, m_list = [], [], [], []

    # usar o maior m como referência espectral
    m_ref = m_values[0]
    x_omp_ref = x_cosamp_ref = x_bp_ref = None

    for row, m in enumerate(m_values):
        if m > N:
            raise ValueError(f"m={m} maior que N={N}")

        indices = np.sort(rng.choice(N, m, replace=False))

        Phi = np.zeros((m, N))
        Phi[np.arange(m), indices] = 1.0

        y = x[indices]
        A = Phi @ D

        kk = min(k, m - 1)

        alpha_omp = omp_reconstruct(A, y, kk)
        alpha_cosamp = cosamp_reconstruct(A, y, kk)
        alpha_bp = bp_reconstruct(A, y, lambda_reg=lambda_reg)

        x_omp = D @ alpha_omp
        x_cosamp = D @ alpha_cosamp
        x_bp = D @ alpha_bp

        rmse_omp_list.append(rmse(x, x_omp))
        rmse_cosamp_list.append(rmse(x, x_cosamp))
        rmse_bp_list.append(rmse(x, x_bp))
        m_list.append(m)

        if m == m_ref:
            x_omp_ref = x_omp.copy()
            x_cosamp_ref = x_cosamp.copy()
            x_bp_ref = x_bp.copy()

        # Coluna 0 – Subamostrado
        ax_sub = axs[row, 0]
        ax_sub.stem(indices, y, basefmt=" ")
        if row == 0:
            ax_sub.set_title("Subamostrado")
        ax_sub.set_ylabel(f"m = {m}")
        ax_sub.grid(True, linestyle="--", alpha=0.3)

        # Coluna 1 – Reconstrução
        ax_rec = axs[row, 1]
        ax_rec.plot(n, x, label="Original", alpha=0.35)
        ax_rec.plot(n, x_omp, "--", label="OMP", linewidth=1.5)
        ax_rec.plot(n, x_cosamp, ":", label="CoSaMP", linewidth=1.2)
        ax_rec.plot(n, x_bp, "-", label="BP (L1)", linewidth=1.2, alpha=0.8)

        if row == 0:
            ax_rec.set_title("Reconstrução: Original vs OMP, CoSaMP e BP")
            ax_rec.legend(fontsize=8, ncol=3)
        if row == len(m_values) - 1:
            ax_rec.set_xlabel("Amostra")

        ax_rec.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # Tabela RMSE
    print("\n========== TABELA DE RMSE NO DOMÍNIO DO TEMPO ==========")
    print(" m  |    OMP        CoSaMP        BP(L1)")
    print("-" * 50)
    for i, m in enumerate(m_list):
        print(f"{m:3d} | {rmse_omp_list[i]:10.3e}  {rmse_cosamp_list[i]:10.3e}  {rmse_bp_list[i]:10.3e}")

    # Tabela espectral (no m_ref)
    if x_omp_ref is not None:
        freqs, F_orig = compute_fft(x, fs)
        _, F_omp = compute_fft(x_omp_ref, fs)
        _, F_cosamp = compute_fft(x_cosamp_ref, fs)
        _, F_bp = compute_fft(x_bp_ref, fs)

        print(f"\n========== ANÁLISE ESPECTRAL (m = {m_ref}) ==========")
        print("H | Freq(Hz) | |X_orig| |X_OMP| Erro% | |X_CoSaMP| Erro% | |X_BP| Erro%")
        print("-" * 95)

        for h in range(1, max_harm + 1, 2):
            freq = h * 60
            idx = int(np.argmin(np.abs(freqs - freq)))

            X0 = F_orig[idx]
            Xo = F_omp[idx]
            Xc = F_cosamp[idx]
            Xb = F_bp[idx]

            def err(a):
                return 0.0 if X0 == 0 else 100.0 * abs(a - X0) / X0

            print(f"{h:2d} | {freq:7.0f} | {X0:7.3f} "
                  f"{Xo:7.3f} {err(Xo):6.1f}% | "
                  f"{Xc:7.3f} {err(Xc):6.1f}% | "
                  f"{Xb:7.3f} {err(Xb):6.1f}%")


def plot_rmse_vs_m(
    x,
    max_m=64,
    min_m=6,
    k=10,
    lambda_reg=5e-5,
    seed=0
):
    """
    Plota RMSE vs m (max_m -> min_m) para OMP, CoSaMP e BP.
    """
    x = np.asarray(x).ravel()
    N = len(x)

    D = dct(np.eye(N), norm="ortho")
    rng = np.random.default_rng(seed)

    rmse_omp_list, rmse_cosamp_list, rmse_bp_list, m_list = [], [], [], []

    for m in range(max_m, min_m - 1, -1):
        if m > N:
            continue

        indices = np.sort(rng.choice(N, m, replace=False))
        Phi = np.zeros((m, N))
        Phi[np.arange(m), indices] = 1.0

        y = x[indices]
        A = Phi @ D

        kk = min(k, m - 1)

        alpha_omp = omp_reconstruct(A, y, kk)
        alpha_cosamp = cosamp_reconstruct(A, y, kk)
        alpha_bp = bp_reconstruct(A, y, lambda_reg=lambda_reg)

        x_omp = D @ alpha_omp
        x_cosamp = D @ alpha_cosamp
        x_bp = D @ alpha_bp

        rmse_omp_list.append(rmse(x, x_omp))
        rmse_cosamp_list.append(rmse(x, x_cosamp))
        rmse_bp_list.append(rmse(x, x_bp))
        m_list.append(m)

    plt.figure(figsize=(10, 6))
    plt.plot(m_list, rmse_omp_list, label="OMP")
    plt.plot(m_list, rmse_cosamp_list, label="CoSaMP")
    plt.plot(m_list, rmse_bp_list, label="BP (L1)")

    plt.gca().invert_xaxis()
    plt.xlabel("Número de amostras m")
    plt.ylabel("RMSE")
    plt.title("RMSE vs m (subamostragem) — OMP, CoSaMP e BP")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_avg_curves(ms, avg_dict, titulo):
    plt.figure(figsize=(10, 5))
    for alg, vals in avg_dict.items():
        plt.plot(ms, vals, label=alg)
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.title(titulo)
    plt.xlabel("m (nº de amostras usadas na reconstrução)")
    plt.ylabel("RMSE médio")
    plt.legend()
    plt.tight_layout()
    plt.show()
