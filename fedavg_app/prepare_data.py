"""Baixa os datasets uma única vez para o volume compartilhado (/data).

    python -m fedavg_app.prepare_data /data              # CIFAR-10
    python -m fedavg_app.prepare_data /data femnist      # FEMNIST
    python -m fedavg_app.prepare_data /data cifar10 femnist
"""

import sys

from fedavg_app.task import DATASETS, download_cifar10, download_femnist

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data"
    names = sys.argv[2:] or ["cifar10"]
    for name in names:
        if name not in DATASETS:
            sys.exit(f"dataset desconhecido: {name!r}. Opções: {DATASETS}")
        {"cifar10": download_cifar10, "femnist": download_femnist}[name](data_dir)
