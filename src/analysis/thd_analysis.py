import numpy as np

# ============================================================
# THD — FUNÇÃO BASE
# ============================================================

def compute_thd(
    signal,
    fs=None,
    fundamental_freq=None,
    max_harmonic=None,
    window=None,
    return_components=False
):
    """
    THD (%) = 100 * sqrt(sum(harmônicos^2)) / fundamental
    Harmônicos considerados: 2..max_harmonic (a fundamental é o 1º harmônico).
    """
    x = np.asarray(signal).ravel()
    N = len(x)

    # Janela opcional (reduz vazamento espectral)
    if window is None:
        xw = x
        scale = 1.0
    else:
        if window == "hann":
            w = np.hanning(N)
        elif window == "hamming":
            w = np.hamming(N)
        else:
            raise ValueError("window deve ser None | 'hann' | 'hamming'")
        xw = x * w
        scale = np.sum(w) / N

    # FFT (meio-espectro) e magnitude normalizada
    X = np.fft.rfft(xw)
    mag = np.abs(X) / (N * scale)

    # Remove DC
    mag[0] = 0.0

    # Bin da fundamental
    if fs is None or fundamental_freq is None:
        k1 = 1  # assume sinal contém 1 período
    else:
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        k1 = int(np.argmin(np.abs(freqs - fundamental_freq)))

    A1 = mag[k1]
    if A1 == 0:
        return np.nan

    if max_harmonic is None:
        max_harmonic = (len(mag) - 1) // max(k1, 1)

    # Harmônicos 2..max_harmonic
    k_end = min(max_harmonic * k1 + 1, len(mag))
    harmonic_amps = mag[2 * k1 : k_end : k1].astype(float)

    thd_ratio = np.sqrt(np.sum(harmonic_amps ** 2)) / A1
    thd_percent = 100.0 * thd_ratio

    if return_components:
        return thd_percent, A1, harmonic_amps

    return thd_percent


# ============================================================
# EXTRAÇÃO DE HARMÔNICAS (1..max_harmonic) — pares e ímpares
# ============================================================

def extract_harmonics(
    signal,
    fs,
    fundamental_freq=60.0,
    max_harmonic=31,
    window=None
):
    """
    Retorna H com amplitudes das harmônicas:
      H[0] = 1ª (fundamental), H[1] = 2ª, ..., H[max_harmonic-1] = max_harmonic
    """
    x = np.asarray(signal).ravel()
    N = len(x)

    # Janela opcional
    if window is None:
        xw = x
        scale = 1.0
    else:
        if window == "hann":
            w = np.hanning(N)
        elif window == "hamming":
            w = np.hamming(N)
        else:
            raise ValueError("window deve ser None | 'hann' | 'hamming'")
        xw = x * w
        scale = np.sum(w) / N

    # FFT e magnitude
    X = np.fft.rfft(xw)
    mag = np.abs(X) / (N * scale)
    mag[0] = 0.0

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    H = np.zeros(max_harmonic, dtype=float)
    for h in range(1, max_harmonic + 1):
        target_f = h * float(fundamental_freq)
        k = int(np.argmin(np.abs(freqs - target_f)))
        H[h - 1] = mag[k] if k < len(mag) else np.nan

    return H


# ============================================================
# ERRO ESPECTRAL POR HARMÔNICA (1..max_harmonic)
# ============================================================

def harmonic_error_per_h(
    original_signal,
    reconstructed_signal,
    fs,
    fundamental_freq=60.0,
    max_harmonic=31,
    window=None,
):
    """
    Retorna vetor E (tamanho max_harmonic):
      E[h-1] = |H_rec(h) - H_orig(h)|   (erro absoluto por harmônica)
    """
    H_orig = extract_harmonics(
        original_signal, fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )
    H_rec = extract_harmonics(
        reconstructed_signal, fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )

    return np.abs(H_rec - H_orig)


def harmonic_nrmse_per_h(
    original_signal,
    reconstructed_signal,
    fs,
    fundamental_freq=60.0,
    max_harmonic=31,
    window=None,
    eps=1e-12,
):
    """
    Retorna vetor N (tamanho max_harmonic):
      N[h-1] = |H_rec(h) - H_orig(h)| / (|H_orig(h)| + eps)

    É uma normalização "por harmônica", ótima pra tabela.
    """
    H_orig = extract_harmonics(
        original_signal, fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )
    H_rec = extract_harmonics(
        reconstructed_signal, fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )

    return np.abs(H_rec - H_orig) / (np.abs(H_orig) + eps)


# ============================================================
# NRMSE ESPECTRAL (agregado no vetor de harmônicas)
# ============================================================

def harmonic_nrmse(
    original_signal,
    reconstructed_signal,
    fs,
    fundamental_freq=60.0,
    max_harmonic=31,
    window=None,
    eps=1e-12
):
    """
    NRMSE no vetor de harmônicas 1..max_harmonic:
      nrmse = RMSE(H_rec - H_orig) / RMS(H_orig)
    """
    H_orig = extract_harmonics(
        original_signal, fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )
    H_rec = extract_harmonics(
        reconstructed_signal, fs=fs,
        fundamental_freq=fundamental_freq,
        max_harmonic=max_harmonic,
        window=window
    )

    num = np.sqrt(np.mean((H_rec - H_orig) ** 2))
    den = np.sqrt(np.mean(H_orig ** 2)) + eps
    return float(num / den)


# ============================================================
# PIPELINE THD (vários sinais)
# ============================================================

def run_thd_analysis(signals, paths, max_harmonic=20, label=""):
    """
    Aplica compute_thd em vários sinais e imprime THD por arquivo + média.
    """
    thd_values = []

    if label:
        print(f"\n=== THD DAS {label.upper()} ===")

    for signal, path in zip(signals, paths):
        thd = compute_thd(signal, max_harmonic=max_harmonic)
        thd_values.append(thd)
        print(f"{path}: THD = {thd:.2f} %")

    if thd_values:
        print(f"THD médio {label}: {np.mean(thd_values):.2f} %")

    return thd_values
