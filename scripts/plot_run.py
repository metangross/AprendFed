#!/usr/bin/env python3
"""Gera os gráficos de uma run, um arquivo por gráfico, numa pasta própria.

Uso:
    python3 scripts/plot_run.py 15926668308773217438     # id completo ou prefixo
    python3 scripts/plot_run.py                          # run mais recente
    python3 scripts/plot_run.py --all                    # todas as runs de outputs/
    python3 scripts/plot_run.py 1592 --format pdf        # PDF ou SVG para o texto

Lê outputs/run-<id>-<data>/metrics.json e clients/*.jsonl e grava em
graficos/run-<id>-<data>/ (a pasta outputs/ pertence ao root dos containers):

    acuracia.png          acurácia do modelo global no teste e média da validação
                          dos clientes, por rodada
    participacao.png      clientes que contribuíram para a agregação em cada rodada
    bateria_clientes.png  todos os clientes lado a lado, na mesma escala
    bateria/cliente-NN.png  um gráfico por cliente
    resumo.csv            tabela por cliente (também impressa no terminal)

Os gráficos de bateria só existem em runs com energy=true; sem bateria inicial,
mostram a energia acumulada.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Paleta de referência (skill de dataviz), modo claro.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # categóricas 1 e 2
CRITICAL = "#d03b3b"  # status: bateria acabou

CONTRIBUTED = ("ok", "partial")


def find_run(arg: str | None) -> Path:
    """Pasta da run pelo id (ou prefixo dele), por um caminho, ou a mais recente."""
    if arg and Path(arg).is_dir():
        return Path(arg)
    outputs = ROOT / "outputs"
    runs = [d for d in outputs.glob("run-*") if (d / "metrics.json").exists()]
    if arg:
        runs = [d for d in runs if d.name.split("-")[1].startswith(arg)]
    if not runs:
        sys.exit(f"Nenhuma run {'com id ' + arg if arg else ''} encontrada em {outputs}")
    ids = {d.name.split("-")[1] for d in runs}
    if len(ids) > 1 and arg:
        sys.exit(f"Prefixo {arg!r} ambíguo: " + ", ".join(sorted(ids)))
    return max(runs, key=lambda d: (d / "metrics.json").stat().st_mtime)


def load_clients(run_dir: Path) -> dict[int, list[dict]]:
    """Linhas de treino de cada cliente, por partição, ordenadas por rodada."""
    clients = {}
    for f in sorted((run_dir / "clients").glob("partition-*.jsonl")):
        rows = [json.loads(line) for line in f.read_text().splitlines() if line]
        train = sorted((r for r in rows if r["phase"] == "train"), key=lambda r: r["round"])
        if train:
            clients[train[0]["partition_id"]] = train
    return dict(sorted(clients.items()))


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def battery_series(rows: list[dict], rounds: int) -> tuple[list[int], list[float]]:
    """Nível da bateria no fim de cada rodada, começando pela carga inicial.

    Rodadas em que o cliente não foi sorteado repetem o nível anterior."""
    by_round = {r["round"]: r for r in rows}
    level = rows[0]["battery_before"]
    xs, ys = [0], [level]
    for rnd in range(1, rounds + 1):
        if rnd in by_round and by_round[rnd].get("battery_after") is not None:
            level = by_round[rnd]["battery_after"]
        xs.append(rnd)
        ys.append(level)
    return xs, ys


def cumulative_energy(rows: list[dict], rounds: int) -> tuple[list[int], list[float]]:
    by_round = {r["round"]: r for r in rows}
    total, xs, ys = 0.0, [0], [0.0]
    for rnd in range(1, rounds + 1):
        total += (by_round.get(rnd) or {}).get("energy_j") or 0.0
        xs.append(rnd)
        ys.append(total)
    return xs, ys


def br(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


DPI = 150


class Run:
    """Dados de uma run já carregados."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.metrics = json.loads((run_dir / "metrics.json").read_text())
        self.rc = self.metrics["run_config"]
        self.rounds = int(self.rc["num-server-rounds"])
        self.clients = load_clients(run_dir)
        self.energy = self.metrics.get("energy", {"enabled": bool(self.rc.get("energy"))})
        self.has_energy = bool(self.energy.get("enabled")) and any(
            "energy_j" in r for rows in self.clients.values() for r in rows
        )
        self.has_battery = self.has_energy and any(
            rows[0].get("battery_before") is not None for rows in self.clients.values()
        )
        fn = battery_series if self.has_battery else cumulative_energy
        self.series = (
            {pid: fn(rows, self.rounds) for pid, rows in self.clients.items()}
            if self.has_energy else {}
        )

    def subtitle(self) -> str:
        rc, m = self.rc, self.metrics
        # Runs anteriores à data no nome da pasta não têm run_date/run_time.
        date = m.get("run_date", "")
        when = f" · {date[:2]}/{date[2:4]}/{date[4:]} {m['run_time']}" if date else ""
        sub = (f"Run {m['run_id']}{when}"
               f" · {len(self.clients)} clientes · {rc.get('model', 'cnn')} · {rc['partitioning']}"
               + (f" (α={rc['dirichlet-alpha']})" if rc["partitioning"] == "dirichlet" else ""))
        if self.has_energy:
            # Runs anteriores à seção "energy" do metrics.json só têm o run_config.
            baseline = self.energy.get("baseline", rc.get("energy-baseline"))
            sub += f" · base {baseline}" if baseline else ""
            sub += f" · bateria vazia={rc['battery-empty-policy']}"
        return sub

    def died(self, pid: int) -> int | None:
        return next((r["round"] for r in self.clients[pid] if r["status"] == "depleted"), None)


def new_figure(width: float, height: float, title: str, subtitles: list[str],
               rows: int = 1, cols: int = 1, extra_top: float = 0.0, **grid):
    """Figura com título e uma linha por subtítulo no topo; alturas em polegadas."""
    top, bottom = 0.55 + 0.2 * len(subtitles) + extra_top, 0.5
    fig = plt.figure(figsize=(width, height + top + bottom), facecolor=SURFACE)
    h = fig.get_figheight()
    fig.text(0.07, 1 - 0.3 / h, title, fontsize=12, weight="bold", color=INK)
    for i, sub in enumerate(subtitles):
        fig.text(0.07, 1 - (0.52 + 0.2 * i) / h, sub, fontsize=8, color=INK_2)
    gs = fig.add_gridspec(rows, cols, left=0.07, right=0.95, top=1 - top / h,
                          bottom=bottom / h, **grid)
    return fig, gs


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_accuracy(run: Run, path: Path) -> None:
    fig, gs = new_figure(10, 3.6, "Acurácia por rodada", [run.subtitle()])
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    m = run.metrics
    srv = {int(k): v["test_acc"] for k, v in m["evaluate_server"].items()}
    cli = {int(k): v["eval_acc"] for k, v in m.get("evaluate_clients", {}).items()
           if "eval_acc" in v}
    for label, data, color in [("teste (modelo global)", srv, SERIES[0]),
                               ("validação dos clientes (média ponderada)", cli, SERIES[1])]:
        if not data:
            continue
        xs = sorted(data)
        ys = [data[x] for x in xs]
        ax.plot(xs, ys, color=color, linewidth=2, solid_capstyle="round",
                solid_joinstyle="round", label=label, zorder=3)
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    if srv:
        last = max(srv)
        ax.annotate(f"{br(100 * srv[last])}%", (last, srv[last]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK,
                    weight="bold")
    ax.set_xlim(0, run.rounds + 0.8)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("rodada", fontsize=8, color=MUTED, labelpad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, path)


def plot_participation(run: Run, path: Path) -> None:
    n = len(run.clients)
    fig, gs = new_figure(10, 2.2, f"Clientes que contribuíram para a agregação (de {n})",
                         [run.subtitle()])
    ax = fig.add_subplot(gs[0])
    style_axes(ax)
    counts = {r: 0 for r in range(1, run.rounds + 1)}
    for rows in run.clients.values():
        for r in rows:
            if r["status"] in CONTRIBUTED and r["round"] in counts:
                counts[r["round"]] += 1
    # Largura da coluna em rodadas, limitada a 24 px.
    px_per_round = ax.get_position().width * fig.get_figwidth() * DPI / (run.rounds + 0.6)
    ax.bar(list(counts), list(counts.values()), width=min(0.6, 24 / px_per_round),
           color=SERIES[0], zorder=3)
    ax.set_xlim(0, run.rounds + 0.6)
    ax.set_ylim(0, max(n, 1))
    ax.set_yticks(sorted({0, n // 2, n}))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("rodada", fontsize=8, color=MUTED, labelpad=2)
    save(fig, path)


def draw_battery(run: Run, ax, pid: int, ymax: float, big: bool = False) -> None:
    """Uma série de bateria (ou energia acumulada) num eixo já criado."""
    style_axes(ax)
    xs, ys = run.series[pid]
    ax.fill_between(xs, ys, color=SERIES[0], alpha=0.10, linewidth=0, zorder=2)
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, solid_capstyle="round", zorder=3)
    died = run.died(pid)
    size = 8.5 if big else 7.5
    if died is not None:
        ax.plot(died, 0, "o", color=CRITICAL, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=5, clip_on=False)
        ax.annotate(f"acabou na r{died}", (died, 0), xytext=(0, 9),
                    textcoords="offset points", fontsize=size, color=INK,
                    ha="right" if died > run.rounds * 0.6 else "left")
    elif big:
        ax.annotate(f"{ys[-1]:.0f} J", (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=size, color=INK,
                    weight="bold")
    ax.set_xlim(0, run.rounds + (0.8 if big else 0))
    ax.set_ylim(0, ymax * 1.08)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10 if big else 4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5 if big else 4))


def client_label(run: Run, pid: int) -> str:
    r0 = run.clients[pid][0]
    return (f"v={br(r0['compute_speed'])} · {r0.get('net_mbps', 0):.0f} Mbps"
            f" · ×{br(r0.get('power_factor', 1.0), 2)} · {r0.get('num_examples', 0)} amostras")


def battery_title(run: Run) -> str:
    return "Bateria restante (J)" if run.has_battery else "Energia acumulada (J)"


def plot_battery_grid(run: Run, path: Path, cols: int) -> None:
    n = len(run.clients)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig, gs = new_figure(2.3 * max(cols, 4) + 0.6, 2.25 * rows - 0.55,
                         f"{battery_title(run)} por cliente", [run.subtitle()],
                         rows=rows, cols=cols, extra_top=0.45, hspace=0.75, wspace=0.18)
    ymax = max(max(ys) for _, ys in run.series.values()) or 1.0
    first = None
    for i, pid in enumerate(run.clients):
        ax = fig.add_subplot(gs[i // cols, i % cols], sharey=first)
        first = first or ax
        draw_battery(run, ax, pid, ymax)
        if i % cols:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_title(f"Cliente {pid}", loc="left", fontsize=9, weight="bold", color=INK,
                     pad=14)
        ax.text(0, 1.02, client_label(run, pid).rsplit(" · ", 1)[0],
                transform=ax.transAxes, fontsize=7, color=INK_2)
    save(fig, path)


def plot_battery_client(run: Run, pid: int, path: Path) -> None:
    # Mesma escala vertical em todos os clientes, para comparar os arquivos entre si.
    ymax = max(max(ys) for _, ys in run.series.values()) or 1.0
    fig, gs = new_figure(10, 3.2, f"{battery_title(run)} · cliente {pid}",
                         [client_label(run, pid), run.subtitle()])
    ax = fig.add_subplot(gs[0])
    draw_battery(run, ax, pid, ymax, big=True)
    ax.set_xlabel("rodada", fontsize=8, color=MUTED, labelpad=2)
    save(fig, path)


def summary(run: Run) -> list[dict]:
    out = []
    for pid, rows in run.clients.items():
        r0 = rows[0]
        contrib = [r["round"] for r in rows if r["status"] in CONTRIBUTED]
        row = {
            "cliente": pid,
            "amostras": r0.get("num_examples"),
            "compute_speed": r0["compute_speed"],
            "net_mbps": r0.get("net_mbps"),
            "rodadas_contribuidas": len(contrib),
            "ultima_rodada": max(contrib) if contrib else None,
        }
        if run.has_energy:
            spent = sum(r.get("energy_j") or 0.0 for r in rows)
            row |= {
                "bateria_inicial_j": r0.get("battery_before"),
                "bateria_final_j": run.series[pid][1][-1] if run.has_battery else None,
                "acabou_na_rodada": run.died(pid),
                "energia_gasta_j": spent,
                "energia_calculo_j": sum(r.get("energy_compute_j") or 0.0 for r in rows),
                "energia_rede_j": sum(r.get("energy_comm_j") or 0.0 for r in rows),
            }
        out.append(row)
    return out


def print_summary(rows: list[dict]) -> None:
    has_energy = "energia_gasta_j" in rows[0]
    head = f"{'cliente':>7} {'amostras':>8} {'v':>5} {'rede':>5} {'contrib.':>8} {'última':>6}"
    if has_energy:
        head += f" {'inicial J':>9} {'final J':>8} {'acabou':>6} {'gasto J':>8} {'cálculo':>7} {'rede':>5}"
    print(head)
    fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
    for r in rows:
        line = (f"{r['cliente']:>7} {fmt(r['amostras'], 'd'):>8} {r['compute_speed']:>5.2f}"
                f" {fmt(r['net_mbps'], '.0f'):>5} {r['rodadas_contribuidas']:>8}"
                f" {fmt(r['ultima_rodada'], 'd'):>6}")
        if has_energy:
            spent = r["energia_gasta_j"] or 1.0
            line += (f" {fmt(r['bateria_inicial_j'], '.1f'):>9}"
                     f" {fmt(r['bateria_final_j'], '.1f'):>8}"
                     f" {fmt(r['acabou_na_rodada'], 'd'):>6} {r['energia_gasta_j']:>8.1f}"
                     f" {100 * r['energia_calculo_j'] / spent:>6.0f}%"
                     f" {100 * r['energia_rede_j'] / spent:>4.0f}%")
        print(line)


def plot_run(run_dir: Path, base: Path, ext: str, cols: int) -> Path:
    run = Run(run_dir)
    out = base / run_dir.name
    plot_accuracy(run, out / f"acuracia.{ext}")
    plot_participation(run, out / f"participacao.{ext}")
    if run.has_energy:
        plot_battery_grid(run, out / f"bateria_clientes.{ext}", cols)
        for pid in run.clients:
            plot_battery_client(run, pid, out / "bateria" / f"cliente-{pid:02d}.{ext}")
    rows = summary(run)
    with open(out / "resumo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", help="id da run (ou prefixo); padrão: a mais recente")
    parser.add_argument("--all", action="store_true", help="todas as runs de outputs/")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "graficos",
                        help="pasta base (padrão: graficos/)")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--cols", type=int, default=5, help="clientes por linha no painel geral")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})

    if args.all:
        runs = sorted(d for d in (ROOT / "outputs").glob("run-*")
                      if (d / "metrics.json").exists() and (d / "clients").is_dir())
        for d in runs:
            try:
                print(f"{d.name} -> {plot_run(d, args.output, args.format, args.cols)}")
            except (KeyError, ValueError) as e:  # runs antigas, de antes de algum campo
                print(f"{d.name}: ignorada ({type(e).__name__}: {e})")
        return

    out = plot_run(find_run(args.run), args.output, args.format, args.cols)
    print(f"Gráficos salvos em {out}/")
    print_summary(summary(Run(find_run(args.run))))


if __name__ == "__main__":
    main()
