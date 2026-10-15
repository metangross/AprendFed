"""Baixa o CIFAR-10 uma única vez para o volume compartilhado (/data)."""

import sys

from fedavg_app.task import download_cifar10

if __name__ == "__main__":
    download_cifar10(sys.argv[1] if len(sys.argv) > 1 else "/data")
