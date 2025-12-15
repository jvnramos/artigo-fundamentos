import numpy as np
import matplotlib.pyplot as plt

def plot_harmonics_from_csv(csv_path, group="CORRENTES", harmonics=(3,5,7), alg="OMP", show_std=True):
    # Lê CSV simples (sem pandas)
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            d = dict(zip(header, parts))
            d["m"] = int(d["m"])
            d["harmonic"] = int(d["harmonic"])
            d["mean"] = float(d["mean"])
            d["std"] = float(d["std"])
            d["n"] = int(d["n"])
            rows.append(d)

    ms = sorted({r["m"] for r in rows if r["group"] == group}, reverse=True)

    plt.figure(figsize=(10,5))

    for h in harmonics:
        means = []
        stds = []
        for m in ms:
            # pega a linha correspondente
            match = [r for r in rows if r["group"]==group and r["m"]==m and r["harmonic"]==h and r["alg"]==alg]
            if not match:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(match[0]["mean"])
                stds.append(match[0]["std"])

        plt.plot(ms, means, label=f"H={h}")
        if show_std:
            lo = np.array(means) - np.array(stds)
            hi = np.array(means) + np.array(stds)
            plt.fill_between(ms, lo, hi, alpha=0.15)

    plt.gca().invert_xaxis()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("m")
    plt.ylabel("Erro médio da harmônica (mean)")
    plt.title(f"{group} — {alg} — erro por harmônica vs m")
    plt.legend()
    plt.tight_layout()
    plt.show()
