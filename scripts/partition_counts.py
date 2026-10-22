#!/usr/bin/env python3
"""Exemplos de treino de cada classe em cada cliente, com a partição do task.py.

Roda no container, que tem os dados (o class_loss.py chama assim):
    docker run --rm -v aprendfed_data:/data:ro -v "$PWD":/app:ro aprendfed:latest \
        python scripts/partition_counts.py '[{"dataset": "cifar10", ...}]'

Cada spec tem dataset, partitioning, alpha, seed, clients e writers_per_client.
Imprime em JSON, para cada spec, a matriz clientes × classes. Conta só a parte
de treino (sem os 20% de validação), como em load_partition.
"""

import json
import sys

import numpy as np

from fedavg_app.task import NUM_CLASSES, _femnist_arrays, femnist_partitions, partition_indices

DATA_DIR = "/data"


def counts(spec: dict) -> list[list[int]]:
    n = spec["clients"]
    if spec["dataset"] == "femnist":
        _, labels, _ = _femnist_arrays(DATA_DIR)
        parts = femnist_partitions(DATA_DIR, n, spec["partitioning"], spec["alpha"],
                                   spec["seed"], spec["writers_per_client"])
    else:
        with np.load(f"{DATA_DIR}/cifar10_train.npz") as d:
            labels = d["y"]
        parts = partition_indices(labels, n, spec["partitioning"], spec["alpha"], spec["seed"])
    out = []
    for idx in parts:
        train = idx[max(1, int(0.2 * len(idx))):]
        out.append(np.bincount(labels[train], minlength=NUM_CLASSES[spec["dataset"]]).tolist())
    return out


if __name__ == "__main__":
    print(json.dumps([counts(s) for s in json.loads(sys.argv[1])]))
