#!/usr/bin/env python3
"""Ganho do leanfed-c sobre o FedAvg contra as classes que o FedAvg perde, por setup.

Uso:
    python3 scripts/plot_gain_classes.py
    python3 scripts/plot_gain_classes.py --ext pdf

Lê o perda_classes.csv (scripts/class_loss.py) de cada setup em SETUPS. Cada
ponto pequeno é uma seed: no eixo x, a fração das classes com mais de 80% dos
exemplos nos clientes que o FedAvg perdeu; no y, o ganho pareado do leanfed-c
(média das últimas rodadas). O ponto grande é a média do setup.

Grava graficos/ganho_classes.<ext>.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_run import (AXIS, INK, INK_2, MUTED, ROOT, SERIES, SURFACE, br,  # noqa: E402
                      new_figure, save, style_axes)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402

# Pasta em graficos/, dataset, rótulo do setup, deslocamento vertical do rótulo (pt).
SETUPS = (
    ("seeds-alfa-0.05", "CIFAR-10", "α = 0,05", 0),
    ("seeds", "CIFAR-10", "α = 0,1", -8),  # à direita há uma seed (40%, +4,4)
    ("seeds-alfa-0.3", "CIFAR-10", "α = 0,3", 0),
    ("seeds-alfa-1.0", "CIFAR-10", "α = 1,0", 7),  # as médias de α = 1,0 e escritor
    ("seeds-femnist-dirichlet", "FEMNIST", "Dirichlet α = 0,1", 0),
    ("seeds-femnist", "FEMNIST", "por escritor", -7),  # quase se sobrepõem
)
COLORS = {"CIFAR-10": SERIES[0], "FEMNIST": SERIES[1]}
# Vazado, por cima: em x = 0 a média do FEMNIST por escritor cobriria a do α = 1,0.
HOLLOW = {"FEMNIST"}


def load(folder: str) -> list[tuple[float, float]]:
    """(classes concentradas no FedAvg, ganho do leanfed-c) de cada seed."""
    with open(ROOT / "graficos" / folder / "perda_classes.csv") as f:
        rows = list(csv.DictReader(f))
    conc = {r["seed"]: float(r["classes_concentradas"]) for r in rows if r["variante"] == "FedAvg"}
    return [(conc[r["seed"]], float(r["ganho"])) for r in rows
            if r["variante"] == "LeanFed-c" and r["seed"] in conc]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ext", default="png", choices=("png", "pdf", "svg"))
    args = parser.parse_args()

    fig, gs = new_figure(
        7.5, 4.2, "O ganho do leanfed-c acompanha as classes que o FedAvg perde",
        ["Cada ponto pequeno é uma seed (42 a 49 no CIFAR-10, 42 a 44 no FEMNIST); o grande, a média do setup.",
         "Ganho = acurácia média das 3 últimas rodadas, pareada por seed.",
         "Perdido = cliente que não contribuiu em nenhuma das 3 últimas rodadas do FedAvg."])
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    ax.axhline(0, color=AXIS, linewidth=1, zorder=1)

    for folder, dataset, label, dy in SETUPS:
        pts = load(folder)
        xs, ys = [p[0] for p in pts], [p[1] * 100 for p in pts]
        color = COLORS[dataset]
        if dataset in HOLLOW:
            seed_style = {"facecolors": "none", "edgecolors": color, "linewidths": 1.2}
            mean_style = {"facecolors": "none", "edgecolors": color, "linewidths": 2, "zorder": 4}
        else:
            seed_style = {"color": color, "edgecolors": SURFACE, "linewidths": 1.5}
            mean_style = {"color": color, "edgecolors": SURFACE, "linewidths": 2, "zorder": 3}
        ax.scatter(xs, ys, s=28, alpha=0.45, zorder=2, **seed_style)
        mx, my = statistics.mean(xs), statistics.mean(ys)
        ax.scatter([mx], [my], s=90, **mean_style)
        ax.annotate(f"{label}  {'+' if my >= 0 else '−'}{br(abs(my))}", (mx, my),
                    xytext=(9, dy), textcoords="offset points",
                    fontsize=8, color=INK, va="center")

    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}".replace("-", "−") if v else "0")
    ax.set_xlim(-0.03, 0.62)
    ax.set_xlabel("Classes com >80% dos exemplos nos clientes que o FedAvg perde",
                  fontsize=9, color=INK_2)
    ax.set_ylabel("Ganho sobre o FedAvg (pontos percentuais)", fontsize=9, color=INK_2)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=7, color=c,
                              markerfacecolor="none" if d in HOLLOW else c,
                              markeredgecolor=c if d in HOLLOW else SURFACE,
                              markeredgewidth=2 if d in HOLLOW else 1, label=d)
                       for d, c in COLORS.items()],
              loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2,
              handletextpad=0.3)
    ax.tick_params(axis="both", colors=MUTED)

    path = ROOT / "graficos" / f"ganho_classes.{args.ext}"
    save(fig, path)
    print(path)


if __name__ == "__main__":
    main()
