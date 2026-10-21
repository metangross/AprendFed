#!/usr/bin/env python3
"""Compara de 2 a 4 runs nos mesmos gráficos.

Uso:
    python3 scripts/compare_runs.py 1977 5123 6789 --labels "FedAvg,LeanFed,LeanFed-c"
    python3 scripts/compare_runs.py 1977 4158 --labels "com bateria,sem energia" -o graficos/x

Grava em graficos/comparacao-<ids>/ (ou -o):

    acuracia_rodada.png   acurácia do modelo global no teste por rodada
    acuracia_energia.png  acurácia × energia acumulada da federação (só runs com energia)
    participacao.png      clientes que contribuíram para a agregação por rodada
    resumo.csv            acurácia final e máxima, energia total, participações e
                          clientes com bateria no fim (também impresso no terminal)

Para a comparação fazer sentido, as runs devem diferir só no que se quer testar
(mesma seed, clientes, baterias e dados); o script avisa quando não é o caso.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_run import (  # noqa: E402
    INK,
    INK_2,
    MUTED,
    ROOT,
    SERIES,
    SURFACE,
    Run,
    br,
    find_run,
    new_figure,
    save,
    style_axes,
)

# Chaves que, se diferentes, tornam a comparação injusta (fora a que se testa).
SAME = ("dataset", "model", "pretrained", "image-size", "partitioning", "dirichlet-alpha",
        "seed", "num-server-rounds", "local-epochs", "batch-size", "lr", "fraction-train")
# Valor das chaves de SAME nas runs anteriores a elas.
SAME_DEFAULTS = {"dataset": "cifar10"}
LABEL_GAP = 0.025  # distância mínima (em acurácia) entre rótulos de ponta


def end_labels(ax, points: list[tuple[float, float, str]]) -> None:
    """Rótulo de valor na ponta de cada linha, omitido quando colidiria com outro;
    a legenda continua identificando todas as séries."""
    placed: list[float] = []
    for x, y, text in sorted(points, key=lambda p: -p[1]):
        if all(abs(y - other) >= LABEL_GAP for other in placed):
            ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK, weight="bold")
            placed.append(y)


def line(ax, xs, ys, color, label) -> None:
    ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
            solid_joinstyle="round", label=label, zorder=3)
    ax.plot(xs[-1], ys[-1], "o", color=color, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)


def subtitle(runs: list[Run]) -> str:
    rc = runs[0].rc
    return (f"{len(runs[0].clients)} clientes · {rc.get('model', 'cnn')} · {rc['partitioning']}"
            + (f" (α={rc['dirichlet-alpha']})" if rc["partitioning"] == "dirichlet" else "")
            + f" · seed {rc['seed']} · runs " + ", ".join(str(r.metrics["run_id"])[:6]
                                                         for r in runs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="ids (ou prefixos) de 2 a 4 runs")
    parser.add_argument("--labels", help="nomes das runs, separados por vírgula")
    parser.add_argument("-o", "--output", type=Path, help="pasta de saída")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    args = parser.parse_args()
    if not 2 <= len(args.runs) <= len(SERIES):
        parser.error(f"compare de 2 a {len(SERIES)} runs")
    labels = args.labels.split(",") if args.labels else args.runs
    if len(labels) != len(args.runs):
        parser.error("--labels deve ter um nome por run")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})

    runs = [Run(find_run(r)) for r in args.runs]
    for key in SAME:
        vals = {str(r.rc.get(key, SAME_DEFAULTS.get(key))) for r in runs}
        if len(vals) > 1:
            print(f"AVISO: as runs diferem em {key}: {sorted(vals)}")
    if len({len(r.clients) for r in runs}) > 1:
        print("AVISO: as runs têm números de clientes diferentes")

    ext = args.format
    out = args.output or ROOT / "graficos" / (
        "comparacao-" + "-".join(str(r.metrics["run_id"])[:6] for r in runs))
    sub = subtitle(runs)
    rounds = max(r.rounds for r in runs)

    # Acurácia por rodada ------------------------------------------------------
    fig, gs = new_figure(10, 3.6, "Acurácia por rodada", [sub])
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    ends = []
    for run, label, color in zip(runs, labels, SERIES):
        acc = run.test_acc()
        xs = sorted(acc)
        line(ax, xs, [acc[x] for x in xs], color, label)
        ends.append((xs[-1], acc[xs[-1]], f"{br(100 * acc[xs[-1]])}%"))
    end_labels(ax, ends)
    ax.set_xlim(0, rounds + 1.2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("rodada", fontsize=8, color=MUTED, labelpad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, out / f"acuracia_rodada.{ext}")

    # Acurácia × energia -------------------------------------------------------
    with_energy = [(r, lb, c) for r, lb, c in zip(runs, labels, SERIES) if r.has_energy]
    if with_energy:
        fig, gs = new_figure(10, 3.6, "Acurácia × energia gasta pela federação", [sub])
        ax = fig.add_subplot(gs[0])
        style_axes(ax)
        xmax, ends = 0.0, []
        for run, label, color in with_energy:
            acc, energy = run.test_acc(), run.energy_by_round()
            rs = [r for r in sorted(acc) if r in energy]
            xs = [energy[r] / 1000 for r in rs]
            line(ax, xs, [acc[r] for r in rs], color, label)
            ends.append((xs[-1], acc[rs[-1]], f"{br(100 * acc[rs[-1]])}%"))
            xmax = max(xmax, xs[-1])
        end_labels(ax, ends)
        ax.set_xlim(0, xmax * 1.1 or 1)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("energia acumulada de todos os clientes (kJ)", fontsize=8,
                      color=MUTED, labelpad=2)
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
        save(fig, out / f"acuracia_energia.{ext}")

    # Participação --------------------------------------------------------------
    n = max(len(r.clients) for r in runs)
    fig, gs = new_figure(10, 2.9, f"Clientes que contribuíram para a agregação (de {n})", [sub])
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    # Colunas agrupadas (uma por run em cada rodada), cada uma com até 24 px e
    # 2 px de folga entre vizinhas; linhas sobrepostas esconderiam runs iguais.
    k = len(runs)
    px_per_round = ax.get_position().width * fig.get_figwidth() * 150 / rounds
    slot = min(0.8 / k, 26 / px_per_round)
    width = slot - 2 / px_per_round
    for i, (run, label, color) in enumerate(zip(runs, labels, SERIES)):
        counts = run.contributed()
        offset = (i - (k - 1) / 2) * slot
        ax.bar([r + offset for r in counts], list(counts.values()), width=width,
               color=color, label=label, zorder=3)
    ax.set_xlim(0.5, rounds + 0.5)
    ax.set_ylim(0, n * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("rodada", fontsize=8, color=MUTED, labelpad=2)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=k, frameon=False,
              fontsize=8, labelcolor=INK_2)
    save(fig, out / f"participacao.{ext}")

    # Resumo ---------------------------------------------------------------------
    rows = []
    for run, label in zip(runs, labels):
        acc = run.test_acc()
        best = max(acc, key=acc.get)
        energy = run.energy_by_round()
        alive = (sum(1 for pid in run.clients if run.series[pid][1][-1] > 0)
                 if run.has_battery else None)
        rows.append({
            "run": label,
            "run_id": run.metrics["run_id"],
            "politica": run.rc.get("energy-policy", "none") if run.has_energy else "sem energia",
            "acuracia_final": acc[max(acc)],
            "acuracia_max": acc[best],
            "rodada_max": best,
            "energia_total_kj": energy[max(energy)] / 1000 if run.has_energy else None,
            "participacoes": sum(run.contributed().values()),
            "clientes_com_bateria_no_fim": alive,
        })
    with open(out / "resumo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Gráficos salvos em {out}/")
    print(f"{'run':<16} {'final':>6} {'máx (rodada)':>13} {'energia kJ':>10} "
          f"{'particip.':>9} {'vivos':>5}")
    for r in rows:
        e = "-" if r["energia_total_kj"] is None else f"{r['energia_total_kj']:.2f}"
        alive = "-" if r["clientes_com_bateria_no_fim"] is None else r["clientes_com_bateria_no_fim"]
        print(f"{r['run'][:16]:<16} {100 * r['acuracia_final']:>5.1f}% "
              f"{100 * r['acuracia_max']:>6.1f}% (r{r['rodada_max']:>2}) {e:>10} "
              f"{r['participacoes']:>9} {alive:>5}")


if __name__ == "__main__":
    main()
