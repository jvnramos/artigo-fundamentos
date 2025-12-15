import os
import numpy as np


def list_locais(base_dir="./dados"):
    # verifica se a pasta base existe
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Pasta base não encontrada: {base_dir}")

    locais = []

    for nome in os.listdir(base_dir):
        caminho = os.path.join(base_dir, nome)

        # ignora arquivos e pastas inválidas
        if not os.path.isdir(caminho):
            continue
        if nome == "__pycache__":
            continue

        # considera o local se tiver tensão ou corrente
        if (
            os.path.isdir(os.path.join(caminho, "tensao")) or
            os.path.isdir(os.path.join(caminho, "corrente"))
        ):
            locais.append(nome)

    return sorted(locais)


def collect_signals(locais, grandeza, fases=("A", "B", "C"), base_dir="./dados"):
    signals = []
    paths = []

    for local in locais:
        for fase in fases:
            # caminho do arquivo esperado
            caminho = os.path.join(
                base_dir,
                local,
                grandeza,
                f"{grandeza}_fase_{fase}.txt"
            )

            # pula se o arquivo não existir
            if not os.path.exists(caminho):
                continue

            # carrega o sinal
            signals.append(np.loadtxt(caminho))
            paths.append(caminho)

    return signals, paths
