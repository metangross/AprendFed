#!/usr/bin/env python3
"""Classes que somem do modelo final porque os clientes que as têm saíram antes do fim.

Uso:
    python3 scripts/class_loss.py                   # setup da run mais recente
    python3 scripts/class_loss.py --ref 1977 -o graficos/seeds
    python3 scripts/class_loss.py --threshold 0.5
    python3 scripts/class_loss.py --bn              # acurácia com a BN recalibrada

Um cliente está perdido se não contribuiu (ok/partial) em nenhuma das --last
últimas rodadas, as mesmas da acuracia_ultimas do seeds_summary.py. Para cada
run do setup (escolhidas como no seeds_summary.py), mede a fração dos exemplos
de treino que estão nos clientes perdidos e a fração das classes com mais de
--threshold dos seus exemplos neles. O ganho é a acuracia_ultimas menos a do
FedAvg com a mesma seed e os mesmos parâmetros que valem para ele (ex.: custo fixo).
Com --bn, a acurácia vem do test_acc_bn, como no seeds_summary.py --bn.

A contagem de classes por cliente precisa dos dados, que estão no volume do
Docker: o script roda o partition_counts.py no container (sem GPU).

Grava perda_classes.csv em graficos/seeds/ (ou -o).
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_run import CONTRIBUTED, ROOT, Run, find_run  # noqa: E402
from seeds_summary import (BUDGET_ONLY, LEANFED_C_ONLY, VARIANT, setup, setup_runs,  # noqa: E402
                           summarize, variant_label)

DOCKER = ["docker", "run", "--rm", "-v", "aprendfed_data:/data:ro",
          "-v", f"{ROOT}:/app:ro", "aprendfed:latest", "python", "scripts/partition_counts.py"]


def partition_spec(run: Run) -> dict:
    rc = run.rc
    return {"dataset": rc.get("dataset", "cifar10"), "partitioning": rc["partitioning"],
            "alpha": rc.get("dirichlet-alpha", 0.5), "seed": rc["seed"],
            "clients": len(run.clients), "writers_per_client": rc.get("writers-per-client", 12)}


def fedavg_key(key: tuple) -> tuple:
    """A variante FedAvg com os mesmos parâmetros que valem para ela (ex.: custo fixo)."""
    v = dict(key, **{"energy-policy": "none"})
    v.update({k: VARIANT[k] for k in LEANFED_C_ONLY + BUDGET_ONLY})
    return tuple(v.items())


def lost_clients(run: Run, from_round: int) -> list[int]:
    return sorted(pid for pid, rows in run.clients.items()
                  if not any(r["round"] >= from_round and r["status"] in CONTRIBUTED
                             for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", help="id (ou prefixo) da run que define o setup")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "graficos" / "seeds")
    parser.add_argument("--last", type=int, default=3,
                        help="rodadas finais em que o cliente precisa contribuir (padrão 3)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="fração dos exemplos de uma classe nos perdidos (padrão 0,8)")
    parser.add_argument("--bn", action="store_true",
                        help="acurácia com a BatchNorm recalibrada (test_acc_bn)")
    args = parser.parse_args()

    metric = "test_acc_bn" if args.bn else "test_acc"
    ref = Run(find_run(args.ref))
    runs = [(key, seed, by_seed[seed][-1])
            for key, by_seed in setup_runs(setup(ref), metric).items()
            for seed in sorted(by_seed)]
    if not runs:
        sys.exit("Nenhuma run com esse setup.")
    # A partição só depende da seed (e do setup): um container conta todas.
    specs = sorted({json.dumps(partition_spec(r), sort_keys=True) for _, _, r in runs})
    out = subprocess.run(DOCKER + ["[" + ",".join(specs) + "]"],
                         capture_output=True, text=True, check=True).stdout
    counts = {s: np.array(c) for s, c in zip(specs, json.loads(out))}

    rows = []
    for key, seed, run in runs:
        c = counts[json.dumps(partition_spec(run), sort_keys=True)]
        lost = lost_clients(run, run.rounds - args.last + 1)
        share = c[lost].sum(axis=0) / c.sum(axis=0)
        rows.append({"variante": variant_label(key), "base": variant_label(fedavg_key(key)),
                     "seed": seed,
                     "run_id": run.metrics["run_id"],
                     "acuracia_ultimas": summarize(run, args.last, metric)["acuracia_ultimas"],
                     "clientes_perdidos": " ".join(map(str, lost)),
                     "dados_perdidos": c[lost].sum() / c.sum(),
                     "classes_concentradas": float(np.mean(share > args.threshold)),
                     "classes": " ".join(map(str, np.flatnonzero(share > args.threshold)))})
    acc = {(r["variante"], r["seed"]): r["acuracia_ultimas"] for r in rows}
    for r in rows:
        base = acc.get((r.pop("base"), r["seed"]))
        r["ganho"] = None if r["variante"].startswith("FedAvg") or base is None \
            else r["acuracia_ultimas"] - base
    rows.sort(key=lambda r: (r["variante"], r["seed"]))

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "perda_classes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Setup de {ref.subtitle()}")
    print(f"perdido = não contribuiu nas {args.last} últimas rodadas; "
          f"concentradas = classes com >{args.threshold:.0%} dos exemplos nos perdidos")
    print(f"\n{'variante':<32} {'seed':>4} {'últimas':>8} {'ganho':>7} {'dados perd.':>11} "
          f"{'concentr.':>9}  clientes perdidos / classes")
    for r in rows:
        ganho = "-" if r["ganho"] is None else f"{r['ganho'] * 100:+.1f}"
        print(f"{r['variante'][:32]:<32} {r['seed']:>4} {r['acuracia_ultimas']:>8.1%} "
              f"{ganho:>7} {r['dados_perdidos']:>11.1%} {r['classes_concentradas']:>9.0%}"
              f"  [{r['clientes_perdidos']}] / [{r['classes']}]")
    for label in sorted({r["variante"] for r in rows}):
        sel = [r for r in rows if r["variante"] == label]
        ganhos = [r["ganho"] for r in sel if r["ganho"] is not None]
        print(f"{label[:32]:<32} {'média':>4} "
              f"{statistics.mean(r['acuracia_ultimas'] for r in sel):>8.1%} "
              f"{(f'{statistics.mean(ganhos) * 100:+.1f}' if ganhos else '-'):>7} "
              f"{statistics.mean(r['dados_perdidos'] for r in sel):>11.1%} "
              f"{statistics.mean(r['classes_concentradas'] for r in sel):>9.0%}")
    print(f"\nCSV salvo em {args.output}/perda_classes.csv")


if __name__ == "__main__":
    main()
