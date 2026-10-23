#!/usr/bin/env python3
"""Acurácia por classe do modelo final de cada run e onde ficam os dados de cada classe.

Roda no container, que tem os dados (sem GPU, a avaliação é leve):
    docker run --rm -v aprendfed_data:/data:ro -v "$PWD":/app:ro aprendfed:latest \
        python scripts/class_accuracy.py outputs/run-<id>-<data> ...

Para cada run imprime, em JSON:
  - acc: acurácia no teste de cada classe, com o final_model.pt (rodada final);
  - acc_check: acurácia global recalculada e a do metrics.json na mesma rodada;
  - share: fração dos exemplos de treino de cada classe nos clientes de cada
    bateria inicial (lida do primeiro registro de treino de cada cliente).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partition_counts import counts  # noqa: E402

from fedavg_app.task import NUM_CLASSES, get_model, load_centralized_test  # noqa: E402

DATA_DIR = "/data"


def initial_battery(run_dir: Path) -> dict[int, float]:
    out = {}
    for f in (run_dir / "clients").glob("partition-*.jsonl"):
        for line in f.open():
            e = json.loads(line)
            if e.get("phase") == "train" and "battery_before" in e:
                out[e["partition_id"]] = e["battery_before"]
                break
    return out


@torch.no_grad()
def class_accuracy(net, loader, num_classes: int) -> tuple[np.ndarray, float]:
    net.eval()
    correct, total = np.zeros(num_classes), np.zeros(num_classes)
    for xb, yb in loader:
        pred = net(xb).argmax(1)
        total += np.bincount(yb.numpy(), minlength=num_classes)
        correct += np.bincount(yb[pred == yb].numpy(), minlength=num_classes)
    return correct / np.maximum(total, 1), correct.sum() / total.sum()


def main() -> None:
    torch.set_num_threads(8)
    loaders = {}
    results = []
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        m = json.loads((run_dir / "metrics.json").read_text())
        rc = m["run_config"]
        dataset = str(rc.get("dataset", "cifar10"))
        nc = NUM_CLASSES[dataset]
        if dataset not in loaders:
            loaders[dataset] = load_centralized_test(DATA_DIR, dataset=dataset)
        net = get_model(str(rc["model"]), False, int(rc["image-size"]), dataset)
        net.load_state_dict(torch.load(run_dir / "final_model.pt", map_location="cpu"))
        acc, overall = class_accuracy(net, loaders[dataset], nc)

        battery = initial_battery(run_dir)
        spec = {"dataset": dataset, "partitioning": rc["partitioning"],
                "alpha": rc.get("dirichlet-alpha", 0.5), "seed": rc["seed"],
                "clients": len(battery), "writers_per_client": rc.get("writers-per-client", 12)}
        per_client = np.array(counts(spec))
        by_tier = defaultdict(lambda: np.zeros(nc))
        for pid, b in battery.items():
            by_tier[b] += per_client[pid]
        totals = per_client.sum(0)
        last = max(m["evaluate_server"], key=int)
        results.append({
            "run_id": m["run_id"], "seed": rc["seed"],
            "acc": acc.tolist(),
            "acc_check": [float(overall), m["evaluate_server"][last]["test_acc"]],
            "share": {str(int(b)): (v / np.maximum(totals, 1)).tolist()
                      for b, v in sorted(by_tier.items())},
        })
        print(f"{run_dir.name}: {overall:.4f}", file=sys.stderr)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
