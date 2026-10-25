#!/usr/bin/env python3
"""Média ± desvio de cada variante sobre as seeds, a partir das runs em outputs/.

Uso:
    python3 scripts/seeds_summary.py                # setup da run mais recente
    python3 scripts/seeds_summary.py --ref 10494    # setup de outra run
    python3 scripts/seeds_summary.py -o graficos/seeds
    python3 scripts/seeds_summary.py --last 5      # média das 5 últimas rodadas
    python3 scripts/seeds_summary.py --bn          # acurácia com a BN recalibrada

Entram as runs com o mesmo setup da run de referência: run_config igual (fora a
seed e as chaves de VARIANT) e as mesmas baterias iniciais, redes e rádios por
cliente.
Elas são agrupadas pela variante (política de energia e seus parâmetros). Se
houver mais de uma run com a mesma variante e seed, vale a mais recente: a mesma
config com a mesma seed dá o mesmo resultado.

Grava resumo_seeds.csv (uma linha por variante) e runs_seeds.csv (uma por run)
em graficos/seeds/ (ou -o). O desvio é o amostral (n − 1).

A acurácia da última rodada oscila muito com dados não-IID (depende de quais
clientes ainda estão vivos); compare as variantes pela média das últimas
rodadas (acuracia_ultimas, --last).

Com --bn, as acurácias vêm do test_acc_bn (BatchNorm recalibrada, runs com
bn-recalibrate=true); as runs sem ele ficam de fora. bn-recalibrate só muda a
medição, então não separa setups: sem --bn, runs com e sem ele se juntam.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_run import ROOT, Run, find_run  # noqa: E402

# Chaves que definem a variante, com o valor das runs anteriores a elas.
VARIANT = {"energy-policy": "none", "comm-estimate": "mean", "eta-after-download": False,
           "min-data-fraction": 0.0, "energy-fixed-j": 0.0, "tail-margin": 0.0,
           "widths": "1.0,0.75,0.5,0.25", "static-bn": False, "width-scaler": False}
# Só valem para o leanfed-c; nas outras políticas não mudam nada.
LEANFED_C_ONLY = ("eta-after-download", "min-data-fraction")
# Valem para as políticas que reservam a comunicação (leanfed-c e width).
COMM_ONLY = ("comm-estimate",)
# Só valem para a política width (HeteroFL).
WIDTH_ONLY = ("widths", "static-bn", "width-scaler")
# Valem para as políticas com orçamento (leanfed e leanfed-c), não para o FedAvg.
BUDGET_ONLY = ("tail-margin",)
IGNORE = ("seed", "min-nodes", "data-dir", "output-dir", "bn-recalibrate")
# Chaves de setup que as runs anteriores a elas não têm, com o valor que tinham.
SETUP_DEFAULTS = {"dataset": "cifar10"}
NAMES = {"none": "FedAvg", "leanfed": "LeanFed", "leanfed-c": "LeanFed-c", "width": "HeteroFL"}
METRICS = (  # coluna, título, formato
    ("acuracia_ultimas", "últimas", "{:.1%}"),
    ("acuracia_final", "final", "{:.1%}"),
    ("acuracia_max", "máx", "{:.1%}"),
    ("energia_total_kj", "energia kJ", "{:.2f}"),
    ("participacoes", "particip.", "{:.1f}"),
    ("rodadas_dead", "dead", "{:.1f}"),
    ("clientes_sem_bateria", "sem bat.", "{:.1f}"),
)


def variant(run: Run) -> tuple:
    v = {k: run.rc.get(k, d) for k, d in VARIANT.items()}
    if v["energy-policy"] != "leanfed-c":
        v.update({k: VARIANT[k] for k in LEANFED_C_ONLY})
    if v["energy-policy"] not in ("leanfed-c", "width"):
        v.update({k: VARIANT[k] for k in COMM_ONLY})
    if v["energy-policy"] != "width":
        v.update({k: VARIANT[k] for k in WIDTH_ONLY})
    if v["energy-policy"] == "none":
        v.update({k: VARIANT[k] for k in BUDGET_ONLY})
    return tuple(v.items())


def variant_label(key: tuple) -> str:
    v = dict(key)
    extra = [f"{k}={v[k]}" for k in VARIANT if k != "energy-policy" and v[k] != VARIANT[k]]
    return " ".join([NAMES.get(v["energy-policy"], v["energy-policy"])] + extra)


def setup(run: Run) -> tuple:
    rc = {k: v for k, v in {**SETUP_DEFAULTS, **run.rc}.items()
          if k not in IGNORE and k not in VARIANT}
    if rc["dataset"] != "femnist":
        rc.pop("writers-per-client", None)
    # radio_w só é gravado desde as runs sem rádio; as anteriores usavam 1,8 W.
    nodes = {pid: (rows[0].get("battery_before"), rows[0].get("net_mbps"),
                   rows[0].get("radio_w", 1.8 if rows[0].get("net_mbps") else None))
             for pid, rows in run.clients.items()}
    return tuple(sorted(rc.items())), tuple(sorted(nodes.items()))


def summarize(run: Run, last: int, metric: str) -> dict:
    acc = run.test_acc(metric)
    rows = [r for rs in run.clients.values() for r in rs]
    return {
        "acuracia_ultimas": statistics.mean(acc[r] for r in sorted(acc)[-last:]),
        "acuracia_final": acc[max(acc)],
        "acuracia_max": max(acc.values()),
        "energia_total_kj": (run.energy_by_round()[run.rounds] / 1000
                             if run.has_energy else None),
        "participacoes": sum(run.contributed().values()),
        "rodadas_dead": sum(r["status"] == "dead" for r in rows),
        "clientes_sem_bateria": (sum(1 for s in run.series.values() if s[1][-1] <= 0)
                                 if run.has_battery else None),
    }


def setup_runs(want: tuple, metric: str) -> dict[tuple, dict[int, list[Run]]]:
    """Runs com o setup `want` e a métrica `metric`, por variante e seed, da mais
    antiga à mais recente."""
    groups: dict[tuple, dict[int, list[Run]]] = {}
    for d in sorted((ROOT / "outputs").glob("run-*")):
        if not (d / "metrics.json").exists():
            continue
        run = Run(d)
        if setup(run) == want and metric in run.metrics["evaluate_server"]["0"]:
            groups.setdefault(variant(run), {}).setdefault(run.rc["seed"], []).append(run)
    for by_seed in groups.values():
        for runs in by_seed.values():
            runs.sort(key=lambda r: (r.dir / "metrics.json").stat().st_mtime)
    return groups


def mean_std(values: list) -> tuple[float | None, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", help="id (ou prefixo) da run que define o setup")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "graficos" / "seeds")
    parser.add_argument("--last", type=int, default=3,
                        help="rodadas finais na média de acuracia_ultimas (padrão 3)")
    parser.add_argument("--bn", action="store_true",
                        help="acurácia com a BatchNorm recalibrada (test_acc_bn)")
    args = parser.parse_args()

    metric = "test_acc_bn" if args.bn else "test_acc"
    ref = Run(find_run(args.ref))
    groups = setup_runs(setup(ref), metric)

    print(f"Setup de {ref.subtitle()}")
    print(f"últimas = acurácia média das {args.last} últimas rodadas"
          + (" (BatchNorm recalibrada)" if args.bn else ""))
    runs_rows, summary_rows = [], []
    for key, by_seed in sorted(groups.items(), key=lambda kv: variant_label(kv[0])):
        label = variant_label(key)
        per_seed = []
        for seed, runs in sorted(by_seed.items()):
            run = runs[-1]
            row = {"variante": label, "seed": seed, "run_id": run.metrics["run_id"],
                   **summarize(run, args.last, metric)}
            if len(runs) > 1:
                print(f"  {label} seed {seed}: {len(runs)} runs, usando {row['run_id']}")
            per_seed.append(row)
        runs_rows += per_seed
        out = {"variante": label, "n": len(per_seed),
               "seeds": " ".join(str(r["seed"]) for r in per_seed)}
        for col, _, _ in METRICS:
            out[col + "_media"], out[col + "_desvio"] = mean_std([r[col] for r in per_seed])
        summary_rows.append(out)

    if not summary_rows:
        sys.exit("Nenhuma run com esse setup.")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("runs_seeds.csv", runs_rows), ("resumo_seeds.csv", summary_rows)):
        with open(args.output / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def cell(m, s, fmt):
        if m is None:
            return "-"
        return fmt.format(m) + ("" if s is None else " ± " + fmt.format(s).rstrip("%"))

    print(f"\n{'variante':<32} {'n':>2} {'seeds':<9} "
          + " ".join(f"{t:>15}" for _, t, _ in METRICS))
    for r in summary_rows:
        print(f"{r['variante'][:32]:<32} {r['n']:>2} {r['seeds']:<9} " + " ".join(
            f"{cell(r[c + '_media'], r[c + '_desvio'], fmt):>15}" for c, _, fmt in METRICS))
    print(f"\nCSVs salvos em {args.output}/")


if __name__ == "__main__":
    main()
