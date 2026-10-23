#!/usr/bin/env python3
"""Gera docs/resultados.pdf (resultados consolidados do AprendFed até 23/10/2026).

Uso:
    python3 docs/fontes/resultados.py [saida.pdf]

Organizado por achado; a ordem dos experimentos e o porquê de cada decisão estão
no historico_projeto.pdf. Requer reportlab, matplotlib e as fontes DejaVu. Os
números vêm de graficos/seeds*/runs_seeds.csv e perda_classes.csv
(scripts/seeds_summary.py e scripts/class_loss.py); estão escritos aqui para o
PDF não depender de outputs/, que não é versionado.
"""

import statistics
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "resultados.pdf"
TMP = Path(tempfile.mkdtemp(prefix="resultados_"))

# ---------------------------------------------------------------- estilo
# Igual ao de historico_projeto.py, para os documentos da pasta combinarem.
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
for name, file in [
    ("DejaVu", "DejaVuSans.ttf"),
    ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
    ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"),
    ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
    ("DejaVuMono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily(
    "DejaVu",
    normal="DejaVu",
    bold="DejaVu-Bold",
    italic="DejaVu-Oblique",
    boldItalic="DejaVu-BoldOblique",
)

BLUE = colors.HexColor("#1F4E8C")
TEXT = colors.HexColor("#222222")
GRAY = colors.HexColor("#6B7280")
NOTE_BG = colors.HexColor("#EDF3FB")
ROW_ALT = colors.HexColor("#F5F7FA")
GRID = colors.HexColor("#D5DCE6")
CODE_BG = colors.HexColor("#F3F4F6")

MARGIN = 0.85 * inch
WIDTH = A4[0] - 2 * MARGIN

S = {
    "title": ParagraphStyle(
        "title", fontName="DejaVu-Bold", fontSize=22, leading=28,
        textColor=BLUE, alignment=TA_CENTER, spaceAfter=10,
    ),
    "sub": ParagraphStyle(
        "sub", fontName="DejaVu", fontSize=11.5, leading=15.5,
        textColor=GRAY, alignment=TA_CENTER, spaceAfter=4,
    ),
    "body": ParagraphStyle(
        "body", fontName="DejaVu", fontSize=10, leading=15,
        textColor=TEXT, spaceAfter=6,
    ),
    "note": ParagraphStyle(
        "note", fontName="DejaVu", fontSize=9.2, leading=13.8, textColor=TEXT,
    ),
    "h1": ParagraphStyle(
        "h1", fontName="DejaVu-Bold", fontSize=17, leading=22, textColor=BLUE,
        spaceBefore=14, spaceAfter=8, keepWithNext=True,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="DejaVu-Bold", fontSize=12.5, leading=16, textColor=BLUE,
        spaceBefore=10, spaceAfter=5, keepWithNext=True,
    ),
    "cell": ParagraphStyle(
        "cell", fontName="DejaVu", fontSize=8.5, leading=12, textColor=TEXT,
    ),
    "head": ParagraphStyle(
        "head", fontName="DejaVu-Bold", fontSize=8.5, leading=12,
        textColor=colors.white,
    ),
    "caption": ParagraphStyle(
        "caption", fontName="DejaVu-Oblique", fontSize=9.2, leading=13.5,
        textColor=TEXT, spaceAfter=6,
    ),
    "code": ParagraphStyle(
        "code", fontName="DejaVuMono", fontSize=7.3, leading=9.6, textColor=TEXT,
    ),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    return P(text, "h1")


def H2(text):
    return P(text, "h2")


def note(text):
    t = Table([[P(text, "note")]], colWidths=[WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 8)])


def table(header, rows, widths, split=False):
    data = [[P(h, "head") for h in header]]
    data += [[P(c, "cell") for c in r] for r in rows]
    ws = [WIDTH * w / sum(widths) for w in widths]
    t = Table(data, colWidths=ws, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    t.setStyle(TableStyle(style))
    if len(rows) < 8 and not split:
        return KeepTogether([t, Spacer(1, 8)])
    t.spaceAfter = 8
    return t


def bullets(items, numbered=False):
    kw = dict(bulletType="1", bulletFormat="%s.") if numbered else dict(
        bulletType="bullet", start="•")
    return ListFlowable(
        [ListItem(P(i), leftIndent=16) for i in items],
        leftIndent=16, bulletFontName="DejaVu", bulletFontSize=8 if not numbered else 9,
        spaceAfter=4, **kw,
    )


def code(text):
    pre = Preformatted(text.strip("\n"), S["code"])
    t = Table([[pre]], colWidths=[WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether([t, Spacer(1, 10)])


def figure(path, width_in, caption):
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    img = Image(str(path), width=width_in * inch, height=width_in * inch * h / w)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 4), img, Spacer(1, 4), P(caption, "caption")])


def footer(canvas, doc):
    canvas.saveState()
    y = 0.55 * inch
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, y + 11, A4[0] - MARGIN, y + 11)
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, y, "Resultados do projeto AprendFed — até 23 de outubro de 2026")
    canvas.drawRightString(A4[0] - MARGIN, y, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------- dados
# Acurácia média no teste por rodada (0 a 20), média das seeds 42 a 49,
# CIFAR-10 com Dirichlet α = 0,1 e D = 0.
ACC_ROUND = {
    "FedAvg": [0.1047, 0.3788, 0.6119, 0.6264, 0.7036, 0.7275, 0.7458, 0.745, 0.7546,
               0.765, 0.7543, 0.7616, 0.7607, 0.7657, 0.7591, 0.7369, 0.7235, 0.7249,
               0.7285, 0.6967, 0.6878],
    "LeanFed": [0.1047, 0.3325, 0.5714, 0.6043, 0.695, 0.7118, 0.7396, 0.7488, 0.7615,
                0.7596, 0.7655, 0.7696, 0.7677, 0.7658, 0.7643, 0.757, 0.7505, 0.748,
                0.7266, 0.7131, 0.6834],
    "LeanFed-c": [0.1047, 0.3008, 0.5246, 0.543, 0.6177, 0.6475, 0.6786, 0.6944, 0.7102,
                  0.7224, 0.733, 0.7415, 0.7483, 0.7494, 0.7553, 0.7555, 0.7571, 0.761,
                  0.7559, 0.7549, 0.7319],
    "LeanFed-c, fração 0,5": [0.1047, 0.299, 0.4899, 0.5199, 0.6197, 0.6396, 0.676, 0.697,
                              0.717, 0.7324, 0.745, 0.7592, 0.7615, 0.7607, 0.7547, 0.7649,
                              0.7776, 0.7844, 0.7807, 0.786, 0.7516],
}
NO_BATTERY_ACC = 0.7855  # mesma configuração sem energia (run 4158…, seed 42)

# Por setup: acurácia média das 3 últimas rodadas do FedAvg e do LeanFed-c, e
# fração das classes com >80% dos exemplos nos clientes que o FedAvg perde (não
# contribuem em nenhuma das 3 últimas rodadas). Seeds 42 a 49 no CIFAR-10, 42 a 44
# no FEMNIST, nessa ordem.
SETUPS = {
    # nome: (dataset, rótulo curto, α no eixo da Figura 2 ou None, FedAvg, LeanFed-c, classes)
    "CIFAR-10, α = 0,05": ("CIFAR-10", "α = 0,05", 0.05,
                            (0.5185, 0.5470, 0.3281, 0.4181, 0.6496, 0.4667, 0.3602, 0.5517),
                            (0.6844, 0.6543, 0.3935, 0.5387, 0.7195, 0.5103, 0.5050, 0.5586),
                            (0.3, 0.2, 0.5, 0.5, 0.4, 0.4, 0.6, 0.2)),
    "CIFAR-10, α = 0,1": ("CIFAR-10", "α = 0,1", 0.1,
                           (0.6961, 0.6739, 0.5836, 0.7726, 0.7399, 0.7107, 0.6941, 0.7640),
                           (0.8043, 0.6938, 0.7134, 0.7545, 0.7810, 0.7279, 0.7516, 0.7539),
                           (0.4, 0.3, 0.4, 0.6, 0.1, 0.2, 0.5, 0.1)),
    "CIFAR-10, α = 0,3": ("CIFAR-10", "α = 0,3", 0.3,
                           (0.8430, 0.8372, 0.8677, 0.8145, 0.8507, 0.8355, 0.8489, 0.8450),
                           (0.8610, 0.8450, 0.8578, 0.8364, 0.8481, 0.8425, 0.8517, 0.8485),
                           (0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0)),
    "CIFAR-10, α = 1,0": ("CIFAR-10", "α = 1,0", 1.0,
                           (0.8801, 0.8811, 0.8730, 0.8727, 0.8833, 0.8748, 0.8668, 0.8772),
                           (0.8752, 0.8731, 0.8671, 0.8663, 0.8823, 0.8705, 0.8741, 0.8692),
                           (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    "FEMNIST, Dirichlet α = 0,1": ("FEMNIST", "Dirichlet α = 0,1", None,
                                    (0.6727, 0.6966, 0.7137), (0.6757, 0.7354, 0.7199),
                                    (0.2258, 0.1935, 0.1774)),
    "FEMNIST, por escritor": ("FEMNIST", "por escritor", None, (0.8006, 0.8238, 0.8344),
                              (0.8022, 0.7979, 0.8336), (0.0, 0.0, 0.0)),
}
# LeanFed-c com fração mínima 0,5, acurácia média das 3 últimas rodadas, seeds 42 a 49.
MDF05 = {
    "CIFAR-10, α = 0,05": (0.7229, 0.6978, 0.5890, 0.6598, 0.7706, 0.5718, 0.5279, 0.6062),
    "CIFAR-10, α = 0,1": (0.7915, 0.7251, 0.7545, 0.7977, 0.8078, 0.7623, 0.7780, 0.7653),
    "CIFAR-10, α = 0,3": (0.8646, 0.8534, 0.8671, 0.8486, 0.8620, 0.8578, 0.8581, 0.8611),
    "CIFAR-10, α = 1,0": (0.8790, 0.8790, 0.8744, 0.8713, 0.8858, 0.8733, 0.8768, 0.8796),
}
# Fração média dos exemplos de treino nos clientes que o FedAvg perde.
DATA_LOST = {"CIFAR-10, α = 0,05": 0.578, "CIFAR-10, α = 0,1": 0.548,
             "CIFAR-10, α = 0,3": 0.450, "CIFAR-10, α = 1,0": 0.406,
             "FEMNIST, Dirichlet α = 0,1": 0.417, "FEMNIST, por escritor": 0.322}
# Quantil 0,975 da t de Student por n (IC 95% do ganho pareado).
T975 = {3: 4.303, 8: 2.365}

# Paleta categórica de scripts/plot_run.py (cores 1 a 4, nesta ordem).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_2, MUTED, AXIS, GRID_HEX = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"


def br(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def signed(x, nd=1):
    return ("+" if x >= 0 else "−") + br(abs(x), nd)


def mean_sd(xs, unit="%"):
    return f"{br(100 * statistics.mean(xs))} ± {br(100 * statistics.stdev(xs))}{unit}"


def gains(name):
    _, _, _, fedavg, leanfed_c, _ = SETUPS[name]
    return [100 * (c - f) for f, c in zip(fedavg, leanfed_c)]


def paired(a, b):
    """Diferença pareada a − b em pontos: texto com média, IC 95% e seeds em que a > b."""
    d = [100 * (x - y) for x, y in zip(a, b)]
    n, m = len(d), statistics.mean(d)
    half = T975[n] * statistics.stdev(d) / n ** 0.5
    return (f"<b>{signed(m)}</b> [{signed(m - half)}; {signed(m + half)}], "
            f"{sum(x > 0 for x in d)} de {n}")


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID_HEX, lw=1)
    ax.set_axisbelow(True)


def figure_accuracy(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    rounds = list(range(21))
    ax.axhline(NO_BATTERY_ACC, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.4, NO_BATTERY_ACC + 0.012, f"sem bateria (seed 42): {br(100 * NO_BATTERY_ACC)}%",
            fontsize=7.5, color=INK_2)
    ends = []
    for (label, ys), color in zip(ACC_ROUND.items(), SERIES):
        ax.plot(rounds, ys, color=color, lw=2, label=label, zorder=3,
                solid_capstyle="round")
        ends.append((ys[-1], f"{label} {br(100 * ys[-1])}%", color))
    # Rótulos diretos na ponta. FedAvg e LeanFed terminam a 0,4 ponto um do outro:
    # os vizinhos próximos se afastam para os dois lados até ficarem a `gap`.
    ends.sort()
    gap = 0.036
    placed = [y for y, _, _ in ends]
    for _ in range(50):
        for i in range(1, len(placed)):
            if placed[i] - placed[i - 1] < gap:
                mid = (placed[i] + placed[i - 1]) / 2
                placed[i - 1], placed[i] = mid - gap / 2, mid + gap / 2
    for (y, text, color), y_text in zip(ends, placed):
        ax.plot(20, y, "o", ms=5, color=color, mec="white", mew=1.2, zorder=4)
        # Fundo branco para o rótulo de cima não ficar riscado pela linha sem bateria.
        ax.text(20.4, y_text, text, fontsize=7.5, color=INK, va="center",
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xlim(0, 26.5)
    ax.set_ylim(0, 0.9)
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_xlabel("rodada", fontsize=8, color=MUTED)
    style_axes(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(0.80, 0.02), frameon=False, fontsize=8,
              labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure_alpha(path: Path) -> None:
    """Acurácia do FedAvg, do LeanFed-c e da fração mínima 0,5 contra o α do CIFAR-10."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    cifar = [(v[2], v[3], v[4], MDF05[k]) for k, v in SETUPS.items() if v[2] is not None]
    alphas = [s[0] for s in cifar]
    series = (("FedAvg", SERIES[0]), ("LeanFed-c", SERIES[2]),
              ("LeanFed-c, fração 0,5", SERIES[3]))
    for i, (label, color) in enumerate(series):
        seeds = [s[1 + i] for s in cifar]
        for a, ys in zip(alphas, seeds):
            ax.scatter([a] * len(ys), ys, s=14, color=color, alpha=0.35, linewidths=0, zorder=2)
        means = [statistics.mean(ys) for ys in seeds]
        ax.plot(alphas, means, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.2,
                label=label, zorder=3)
        ax.text(alphas[0] / 1.12, means[0], f"{label} {br(100 * means[0])}%", fontsize=7.5,
                color=INK, va="center", ha="right")
    # Ganho sobre o FedAvg, uma linha por método, na cor da série.
    for a, f, *others in cifar:
        for (label, color), ys, y in zip(series[1:], others, (0.975, 0.94)):
            g = 100 * (statistics.mean(ys) - statistics.mean(f))
            ax.text(a, y, signed(g), fontsize=7.5, color=color, ha="center",
                    fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(alphas)
    ax.set_xticklabels([br(a, 2 if a < 0.1 else 1) for a in alphas])
    ax.minorticks_off()
    ax.set_xlim(0.022, 1.4)
    ax.set_ylim(0.3, 1.0)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([f"{v}%" for v in range(30, 100, 10)])
    ax.set_xlabel("α da partição Dirichlet (escala log; menor = classes mais concentradas)",
                  fontsize=8, color=MUTED)
    style_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def figure_gain_classes(path: Path) -> None:
    """Ganho do LeanFed-c contra as classes concentradas, uma marca por seed e por setup."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    color = SERIES[2]
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    # Perto de x = 0, α = 0,3, α = 1,0 e escritor ficam a menos de 1,5 ponto um do outro,
    # e os rótulos do FEMNIST Dirichlet e do α = 0,1 passariam sobre seeds: esses vão
    # para áreas vazias, com linha de chamada.
    callout = {"CIFAR-10, α = 0,3": (0.12, -1.0), "CIFAR-10, α = 1,0": (0.12, -2.1),
               "FEMNIST, por escritor": (0.12, -3.2),
               "FEMNIST, Dirichlet α = 0,1": (0.03, 7.5),
               "CIFAR-10, α = 0,1": (0.53, 2.0)}
    for name, (dataset, label, _, _, _, conc) in SETUPS.items():
        g = gains(name)
        # FEMNIST vazado e por cima, para não cobrir a média do α = 1,0 em x = 0.
        hollow = dataset == "FEMNIST"
        face = "none" if hollow else color
        edge = color if hollow else "white"
        ax.scatter(conc, g, s=22, facecolors=face, edgecolors=edge, alpha=0.45,
                   linewidths=1.1, zorder=2)
        mx, my = statistics.mean(conc), statistics.mean(g)
        ax.scatter([mx], [my], s=75, facecolors=face, edgecolors=edge, linewidths=1.8,
                   zorder=4 if hollow else 3)
        text = f"{dataset} {label}  {signed(my)}"
        if name in callout:
            ax.annotate(text, (mx, my), xytext=callout[name], fontsize=7.5, color=INK,
                        va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                                     relpos=(0, 0.5), shrinkA=3, shrinkB=5))
        else:
            # α = 0,05: logo à direita da média há uma seed (40%, +7,0).
            dy = 8 if name == "CIFAR-10, α = 0,05" else 0
            ax.annotate(text, (mx, my), xytext=(8, dy), textcoords="offset points",
                        fontsize=7.5, color=INK, va="center")
    ax.set_xlim(-0.03, 0.75)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticklabels([f"{v}%" for v in range(0, 70, 10)])
    ax.set_yticks(range(-4, 20, 4))
    ax.set_yticklabels([signed(v, 0) if v else "0" for v in range(-4, 20, 4)])
    ax.set_xlabel("classes com >80% dos exemplos nos clientes que o FedAvg perde",
                  fontsize=8, color=MUTED)
    ax.set_ylabel("ganho sobre o FedAvg (pontos)", fontsize=8, color=MUTED)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- conteúdo
def setup_rows():
    rows = []
    for name, (_, _, _, fedavg, leanfed_c, conc) in SETUPS.items():
        g = gains(name)
        n, m = len(g), statistics.mean(g)
        half = T975[n] * statistics.stdev(g) / n ** 0.5
        rows.append([name, str(n), mean_sd(fedavg, ""), mean_sd(leanfed_c, ""),
                     f"<b>{signed(m)}</b> [{signed(m - half)}; {signed(m + half)}]",
                     f"{sum(x > 0 for x in g)} de {n}",
                     f"{br(100 * DATA_LOST[name], 0)}%",
                     f"{br(100 * statistics.mean(conc), 0)}%"])
    return rows


def mdf_rows():
    rows = []
    for name, mdf in MDF05.items():
        _, label, _, fedavg, leanfed_c, _ = SETUPS[name]
        rows.append([label.removeprefix("α = "), mean_sd(fedavg, "%"), mean_sd(leanfed_c, "%"), mean_sd(mdf, "%"),
                     paired(mdf, leanfed_c), paired(mdf, fedavg)])
    return rows


def build() -> list:
    s = []
    s += [
        Spacer(1, 30),
        P("Resultados do AprendFed", "title"),
        P("LeanFed-c contra FedAvg com bateria limitada: o que os experimentos mostram",
          "sub"),
        P("Até 23 de outubro de 2026 — CIFAR-10 com 8 seeds, FEMNIST com 3 — branch "
          "<i>leanfed-baseline</i> (commit 851ac0d)",
          "sub"),
        Spacer(1, 12),
        note("<b>Resumo.</b> Com bateria simulada e custo de comunicação realista, o "
             "LeanFed original <b>não melhora</b> sobre o FedAvg (+0,3 ponto com 8 seeds). A "
             "versão corrigida (<b>LeanFed-c</b>), que reserva a comunicação e o custo fixo "
             "antes de escolher quantos dados treinar, mantém praticamente todos os clientes "
             "vivos até o fim. Com 8 seeds no CIFAR-10, isso vale <b>+9,1 pontos</b> de "
             "acurácia com Dirichlet α = 0,05 (positivo nas 8 seeds) e <b>+4,3</b> com "
             "α = 0,1 (6 de 8; IC 95% de −0,1 a +8,8). Quando os dados de cada classe estão "
             "espalhados entre os clientes, o ganho some: +0,6 com α = 0,3 e −0,4 com "
             "α = 1,0. Pular as rodadas em que o cliente treinaria menos da metade dos dados "
             "(<b>fração mínima 0,5</b>) melhora o LeanFed-c <b>nos quatro α</b>, de +7,3 "
             "pontos (α = 0,05) a +0,5 (α = 1,0): sobre o FedAvg, ganha +16,3, +6,8 e +1,6 "
             "de α = 0,05 a α = 0,3 e empata com α = 1,0, onde o LeanFed-c sozinho perdia. "
             "Funciona porque os clientes de bateria pequena pulam o começo do treino e "
             "chegam ao fim com energia para treinar todos os dados; quanto maior a fração "
             "mínima, mais forte o efeito (com 1,0, +8,5 sobre o FedAvg em α = 0,1). Entre "
             "cenários, o ganho acompanha a fração de classes concentradas nos clientes que "
             "o FedAvg perde por falta de bateria, e não a quantidade de dados que esses "
             "clientes levam; seed a seed, essa fração não prevê o ganho."),
        P("<b>Conteúdo</b>"),
        bullets([
            "1. Cenário, políticas e métricas",
            "2. O LeanFed-c ganha porque não perde clientes no fim",
            "3. Ajustes do LeanFed-c: a fração mínima ajuda em todos os α",
            "4. O ganho depende da concentração de classes",
            "5. Conclusões, limitações e próximos passos",
            "Apêndice A. Runs e como reproduzir",
        ]),
        P("A ordem dos experimentos e o motivo de cada decisão estão em "
          "<i>historico_projeto.pdf</i>, na mesma pasta."),
    ]

    # 1 ----------------------------------------------------------------
    s += [
        H1("1. Cenário, políticas e métricas"),
        P("<b>Cenário de referência.</b> FedAvg com Flower, um container Docker por "
          "cliente. 15 clientes, baterias de 1500/1000/500 J (5 clientes de cada), rede de "
          "25 Mbps, MobileNetV3-Small pré-treinada a 96 px, 20 rodadas, uma época local, "
          "todos os clientes sorteados em toda rodada. Dataset padrão: CIFAR-10 dividido "
          "por Dirichlet α = 0,1."),
        P("<b>Energia.</b> O treino custa 8,88 mJ por amostra (perfil medido na GPU com "
          "NVML, acima do repouso). A comunicação é potência do rádio (1,8 W) × bits / "
          "vazão, com a vazão sorteada em U(1, 25) Mbps a cada rodada: em média 23,6 J por "
          "rodada, com cauda longa (quase 100 J com vazão perto de 1 Mbps). Quando a "
          "bateria acaba, o cliente morre no meio da rodada."),
        table(
            ["Política", "Fração η dos dados treinada por rodada", "Ideia"],
            [
                ["FedAvg", "1", "treina tudo até a bateria acabar"],
                ["LeanFed", "orçamento / A",
                 "LeanFed original: supõe que só o treino gasta energia"],
                ["LeanFed-c", "(orçamento − C − D) / A",
                 "LeanFed corrigido: reserva antes a rede (C) e o custo fixo (D)"],
            ],
            [1.0, 1.8, 3],
        ),
        P("O orçamento da rodada é a bateria atual dividida pelas participações que "
          "faltam; A é a energia de treinar todos os dados locais. Com η = 0, o cliente "
          "pula a rodada sem gastar nada."),
        P("<b>Métricas.</b>"),
        bullets([
            "<b>Acurácia</b>: média da acurácia no teste nas <b>3 últimas rodadas</b>. A "
            "da última rodada sozinha oscila vários pontos conforme quais clientes ainda "
            "estão vivos.",
            "<b>Ganho</b>: acurácia do LeanFed-c menos a do FedAvg <b>com a mesma seed</b> "
            "(mesma partição e mesma sequência de vazões), com intervalo de confiança de "
            "95% pela t de Student sobre as diferenças pareadas.",
            "<b>Seeds</b>: valores são média ± desvio amostral. No CIFAR-10, as políticas "
            "e as variantes da seção 3 têm as seeds 42 a 49 (n = 8); o FEMNIST tem só as "
            "seeds 42, 43 e 44 (n = 3). Diferenças entre duas variantes do LeanFed-c também "
            "são pareadas por seed.",
            "<b>Cliente perdido</b>: não contribuiu em nenhuma das 3 últimas rodadas. "
            "<b>Classes concentradas</b>: fração das classes com mais de 80% dos seus "
            "exemplos de treino nos clientes perdidos pelo FedAvg.",
        ]),
    ]

    # 2 ----------------------------------------------------------------
    s += [
        H1("2. O LeanFed-c ganha porque não perde clientes no fim"),
        table(
            ["CIFAR-10, α = 0,1, seeds 42 a 49", "Média das 3 últimas", "Acurácia máxima",
             "Rodadas <i>dead</i>", "Sem bateria no fim"],
            [
                ["FedAvg", "70,4 ± 6,0%", "78,2 ± 1,4%", "45,5", "7,8"],
                ["LeanFed", "70,8 ± 6,5%", "77,6 ± 1,6%", "29,4", "7,8"],
                ["LeanFed-c", "<b>74,8 ± 3,6%</b>", "77,5 ± 2,5%", "0,8", "1,6"],
            ],
            [2.2, 1.5, 1.4, 1.1, 1.2],
        ),
        figure(TMP / "acuracia.png", 6.3,
               "Figura 1. Acurácia no teste por rodada, média das seeds 42 a 49 (CIFAR-10, "
               "α = 0,1). FedAvg, LeanFed e LeanFed-c chegam a picos parecidos (76,6%, 77,0% "
               "e 76,1%). A diferença está no fim, quando os clientes de bateria pequena "
               "morrem no FedAvg e no LeanFed. O LeanFed-c com fração mínima 0,5 (seção 3) "
               "chega a um pico maior (78,6%). Todas as curvas caem na rodada 20."),
        bullets([
            "<b>O LeanFed-c ganha em 6 das 8 seeds</b>: +10,8, +2,0 e +13,0 pontos nas "
            "seeds 42 a 44; −1,8, +4,1, +1,7, +5,8 e −1,0 nas seeds 45 a 49. A média cai de "
            "+8,6 (n = 3) para <b>+4,3</b>, com IC 95% de −0,1 a +8,8 (t pareado, "
            "p = 0,055).",
            "<b>O pico não melhora</b>: 77,5% no LeanFed-c contra 78,2% no FedAvg. O ganho "
            "vem de não perder acurácia nas últimas rodadas, e não de aprender mais.",
            "<b>O LeanFed original não ajuda</b>: +0,3 ponto sobre o FedAvg (IC 95% de −0,6 "
            "a +1,3). Gasta o orçamento todo em treino, paga a rede por fora e termina com "
            "os mesmos 7,8 clientes sem bateria do FedAvg. Morre menos rodadas no meio do "
            "caminho (29 contra 45), mas isso não muda a acurácia. Aqui a comunicação é de "
            "um terço a metade do custo de uma rodada.",
            "<b>A energia total é a mesma</b> (11,3 kJ no FedAvg, 11,2 no LeanFed-c): o "
            "LeanFed-c distribui a bateria ao longo das 20 rodadas, com 285 participações "
            "contra 247 do FedAvg.",
        ]),
        P("<b>Por que sobram perdas.</b> Na seed 43 (ganho de só 2 pontos), três clientes "
          "do LeanFed-c morrem nas rodadas 18 a 20, cada um por <b>uma</b> rodada de vazão "
          "perto de 1 Mbps (52 a 127 J de rede contra 23,6 J reservados), quando a bateria "
          "já não tinha folga. Dois deles tinham 94% dos pássaros e 92% dos cervos, que "
          "caem a 48% e 18% de acurácia no modelo final."),
        P("<b>Quando o LeanFed-c perde.</b> Nas seeds 45 e 49, o FedAvg quase não cai no "
          "fim: na 49, a última rodada é o seu máximo (78,0%); na 45, ele fica em 77,3% "
          "nas 3 últimas, embora perca 74% dos dados e 60% das classes fiquem "
          "concentradas nos clientes que saem. O LeanFed-c, que treina só parte dos dados "
          "por rodada, chega a picos menores (79,6% e 77,8% contra 81,0% e 78,0%) e, na "
          "seed 45, ainda perde 2 clientes no fim. Com a fração mínima 0,5 (seção 3), as "
          "duas seeds passam a ganhar (+2,5 e +0,1)."),
    ]

    # 3 ----------------------------------------------------------------
    s += [
        H1("3. Ajustes do LeanFed-c: a fração mínima ajuda em todos os α"),
        P("A <b>fração mínima</b> faz o cliente pular a rodada quando a fração η que cabe "
          "no orçamento fica abaixo do mínimo. A energia não gasta passa para as rodadas "
          "seguintes. Na última participação, o mínimo não vale. Com fração mínima 1,0, o "
          "cliente treina todos os dados ou pula."),
        table(
            ["Variante (α = 0,1, 8 seeds)", "Média das 3 últimas",
             "Diferença para o LeanFed-c [IC 95%], seeds melhores", "Acurácia máxima",
             "Particip.", "Rodadas <i>dead</i>"],
            [
                ["LeanFed-c", "74,8 ± 3,6%", "—", "77,5 ± 2,5%", "285", "0,8"],
                ["fração mínima 0,3", "76,6 ± 2,2%", "<b>+1,9</b> [+0,6; +3,2], 7 de 8",
                 "79,4 ± 2,1%", "270", "0,4"],
                ["fração mínima 0,5", "77,3 ± 2,7%", "<b>+2,5</b> [+1,0; +4,1], 7 de 8",
                 "80,1 ± 1,9%", "260", "0,4"],
                ["fração mínima 0,7", "77,6 ± 3,4%", "<b>+2,9</b> [+1,9; +3,8], 8 de 8",
                 "80,0 ± 1,9%", "251", "0,4"],
                ["fração mínima 1,0", "<b>78,9 ± 3,1%</b>", "<b>+4,1</b> [+1,9; +6,4], 8 de 8",
                 "80,0 ± 2,9%", "240", "0"],
                ["margem de cauda 0,9", "75,4 ± 3,6%", "+0,6 [−0,8; +2,1], 4 de 8",
                 "77,3 ± 2,9%", "278", "0"],
            ],
            [1.75, 1.2, 2.1, 1.1, 0.9, 0.95],
        ),
        bullets([
            "<b>A fração mínima é o único ajuste que melhora a acurácia</b>, e o ganho "
            "cresce com o mínimo: +1,9, +2,5, +2,9 e +4,1 pontos sobre o LeanFed-c com 0,3, "
            "0,5, 0,7 e 1,0 (t pareado, p de 0,012 a &lt; 0,001). Com 0,5, o efeito é o mesmo "
            "nas seeds 42 a 44 (+2,0) e nas 45 a 49 (+2,8). Com 3 seeds, esse ganho parecia "
            "ruído.",
            "<b>Treinar uma fração dos dados não ajuda.</b> A melhor variante é a 1,0, que "
            "treina tudo ou pula: 78,9%, nenhuma rodada <i>dead</i> e +8,5 pontos sobre o "
            "FedAvg (IC 95% de +5,3 a +11,7; 8 de 8). A diferença para a 0,5 (+1,6; IC 95% "
            "de −0,3 a +3,6; p = 0,09) não é significativa, mas não há nenhum sinal a favor "
            "de frações menores. O que ajuda é pular rodadas para acumular energia. A fração "
            "0,5 segue como valor de referência, e a varredura entra como análise de "
            "sensibilidade.",
            "<b>Contra o FedAvg, a fração mínima 0,5 ganha nas 8 seeds</b>: +6,8 pontos "
            "(IC 95% de +2,6 a +11,1; p = 0,007), contra +4,3 do LeanFed-c sem ela. As seeds "
            "45 e 49, em que o LeanFed-c perdia, ficam em +2,5 e +0,1.",
            "<b>Diferente do LeanFed-c, a fração mínima também sobe o pico</b>: 80,1% "
            "contra 78,2% do FedAvg com 0,5 (+1,8 ponto pareado, 7 de 8 seeds, p = 0,002); "
            "80,0% com 0,7 e 1,0.",
            "<b>Margem de cauda</b> (guardar o p90 da comunicação até a última "
            "participação): zera as mortes, mas não muda a acurácia (+0,6 ponto, 4 de 8 "
            "seeds). Depois de uma rodada de rede ruim no fim, o cliente pula as últimas "
            "rodadas em vez de morrer, e as classes dele somem do modelo do mesmo jeito.",
            "<b>Reserva por percentil</b> em vez da média (seed 42): o p90 evita mortes mas "
            "corta 47 participações; o p50 volta a perder clientes. A média ficou como "
            "padrão.",
        ]),
        P("<b>Custo fixo por rodada.</b> Com D = 5 J (D ≈ C/5, como no exemplo do documento "
          "de custo de energia), FedAvg e LeanFed perdem cerca de 2,5 pontos, piores nas 8 "
          "seeds (67,9 ± 6,3% e 68,4 ± 6,3%), e o LeanFed morre mais (29 → 42 rodadas "
          "<i>dead</i>). O LeanFed-c, que já reserva D, fica igual (74,2 ± 3,3%; −0,5 "
          "ponto, não significativo). A vantagem dele sobre o FedAvg sobe para "
          "<b>+6,3 pontos</b> (IC 95% de +0,8 a +11,8; 6 de 8 seeds). Com 3 seeds, parecia "
          "de 10."),
        H2("Por que a fração mínima funciona"),
        P("Os logs dos clientes mostram quando os 5 clientes de 500 J treinam (α = 0,1, "
          "média das 8 seeds). Os de 1000 e 1500 J participam de quase todas as rodadas em "
          "todas as políticas."),
        table(
            ["Clientes de 500 J", "Ativos nas rodadas 1 a 7", "Ativos nas rodadas 18 a 20",
             "Exemplos treinados por rodada, rodadas 18 a 20"],
            [
                ["FedAvg", "98%", "8%", "0,1 mil"],
                ["LeanFed-c", "91%", "78%", "3,8 mil"],
                ["fração mínima 0,5", "31%", "85%", "8,1 mil"],
                ["fração mínima 0,7", "23%", "85%", "9,4 mil"],
                ["fração mínima 1,0", "17%", "94%", "11,5 mil"],
            ],
            [1.5, 1.2, 1.2, 1.8],
        ),
        bullets([
            "O FedAvg gasta a bateria dos clientes de 500 J no começo e chega ao fim sem "
            "eles. O LeanFed-c os mantém vivos, mas cada um treina pouco por rodada (no "
            "fim, 3,8 mil exemplos somados entre os 5).",
            "Com a fração mínima, esses clientes pulam o começo, acumulam energia e treinam "
            "quase todos os dados nas últimas rodadas. As classes deles pesam mais nos "
            "modelos dessas rodadas, que são os que a métrica mede. Quanto maior o mínimo, mais a "
            "participação se concentra no fim, na mesma ordem do ganho da tabela acima.",
            "O mecanismo é o oposto de fazer os clientes frágeis contribuírem mais cedo: "
            "a fração mínima ganha porque eles contribuem <b>tarde</b>, e com mais dados.",
        ]),
        H2("A fração mínima 0,5 nos outros α"),
        table(
            ["α", "FedAvg", "LeanFed-c", "Fração mínima 0,5",
             "Fração 0,5 − LeanFed-c [IC 95%], seeds melhores",
             "Fração 0,5 − FedAvg [IC 95%], seeds melhores"],
            mdf_rows(),
            [0.75, 1.2, 1.2, 1.1, 1.65, 1.75],
        ),
        bullets([
            "<b>Melhora o LeanFed-c em todos os α</b>, em 8 de 8 seeds em α = 0,05, 0,3 e "
            "1,0. A diferença diminui quando as classes se espalham, como o próprio ganho do "
            "LeanFed-c (seção 4).",
            "<b>Com α = 0,05, o ganho sobre o FedAvg quase dobra</b>: +16,3 pontos contra "
            "+9,1 do LeanFed-c sozinho. O pico também passa a ser maior que o do FedAvg "
            "(+3,7 pontos, 8 de 8); sem a fração mínima, o LeanFed-c ficava 4,2 pontos "
            "abaixo (0 de 8).",
            "<b>Com α = 0,3</b>, o ganho sobre o FedAvg passa a ser significativo: +1,6 "
            "pontos (7 de 8, p = 0,003), contra +0,6 sem significância do LeanFed-c.",
            "<b>Com α = 1,0, elimina o custo do LeanFed-c</b>: empata com o FedAvg (+0,1, "
            "4 de 8; pico 88,0% nos dois), onde o LeanFed-c ficava 0,4 ponto abaixo.",
        ]),
        note("<b>O que só foi medido no α = 0,1.</b> As frações 0,3, 0,7 e 1,0, a margem de "
             "cauda e o custo fixo D = 5. Nenhuma variante foi rodada no FEMNIST."),
    ]

    # 4 ----------------------------------------------------------------
    s += [
        H1("4. O ganho depende da concentração de classes"),
        P("<b>Pergunta.</b> O LeanFed-c mantém os clientes vivos em todos os cenários. "
          "Quando isso vale acurácia? Foram testados dois tipos de cenário, cada um com "
          "FedAvg e LeanFed-c (8 seeds no CIFAR-10, 3 no FEMNIST):"),
        bullets([
            "<b>FEMNIST</b> (caracteres escritos à mão, 62 classes, 12 pessoas por "
            "cliente), dividido <b>por escritor</b>, que é a partição natural, ou por "
            "<b>Dirichlet α = 0,1</b>, com as mesmas pessoas.",
            "<b>CIFAR-10 variando o α</b> da partição Dirichlet: 0,05, 0,1, 0,3 e 1,0. Só o "
            "α muda; quanto menor, mais cada classe fica concentrada em poucos clientes.",
        ]),
        table(
            ["Cenário", "n", "FedAvg (%)", "LeanFed-c (%)", "Ganho [IC 95%]",
             "Seeds com ganho", "Dados perdidos pelo FedAvg", "Classes com &gt;80% perdidas"],
            setup_rows(),
            [1.55, 0.4, 1.1, 1.1, 1.6, 0.85, 1.0, 1.05],
        ),
        figure(TMP / "alfa.png", 6.3,
               "Figura 2. Acurácia média das 3 últimas rodadas contra o α do CIFAR-10, "
               "8 seeds por ponto. Os pontos claros são as seeds; no alto, o ganho médio "
               "sobre o FedAvg, na cor de cada método. Todos pioram quando o α diminui; o "
               "FedAvg piora mais. A fração mínima 0,5 (seção 3) fica acima do LeanFed-c "
               "em todos os α."),
        figure(TMP / "ganho_classes.png", 6.3,
               "Figura 3. Ganho do LeanFed-c contra a fração de classes concentradas nos "
               "clientes que o FedAvg perde. Cada ponto pequeno é uma seed (8 no CIFAR-10, "
               "3 no FEMNIST); o grande, a média do cenário. CIFAR-10 preenchido, FEMNIST "
               "vazado."),
        P("<b>O que os números dizem.</b>"),
        bullets([
            "<b>Perder dados não custa; perder classes custa.</b> O FedAvg perde de 32% a "
            "58% dos dados de treino em todos os cenários, mas só fica atrás quando as "
            "classes estão concentradas nos clientes que saem. Com α = 0,3, ele perde 45% "
            "dos dados e 5% das classes, e o ganho do LeanFed-c é de 0,6 ponto, sem "
            "significância (IC 95% de −0,3 a +1,5).",
            "<b>O resultado se repete no mesmo dataset.</b> A comparação entre CIFAR-10 e "
            "FEMNIST mudava o dataset junto com a partição. A varredura do α muda só a "
            "concentração e reproduz a mesma relação: +9,1, +4,3, +0,6 e −0,4 pontos de "
            "α = 0,05 a α = 1,0.",
            "<b>Só α = 0,05 tem ganho firme.</b> Positivo nas 8 seeds (de +0,7 a +16,6; "
            "t pareado p = 0,002). Em α = 0,1 o intervalo encosta no zero.",
            "<b>Sem concentração, o LeanFed-c custa pouco.</b> Com α = 1,0 ele fica abaixo "
            "do FedAvg em 7 das 8 seeds (−0,4 ponto, IC 95% de −0,8 a 0,0), e a acurácia "
            "máxima também é menor (87,4% contra 88,0%). Sem classes a proteger, sobra só o "
            "custo de treinar menos dados por rodada. Com a fração mínima 0,5, esse custo "
            "some (seção 3).",
            "<b>O ganho é relativo.</b> Com α menor os dois métodos pioram (Figura 2). O "
            "LeanFed-c não fica melhor com dados mais heterogêneos; ele resiste melhor a um "
            "cenário pior.",
            "<b>A relação vale entre cenários, não seed a seed.</b> Dentro de um mesmo α, a "
            "fração de classes concentradas não prevê o ganho (correlação de Spearman 0,30 "
            "com α = 0,05 e 0,14 com α = 0,1, ambas não significativas). Na seed 45 com "
            "α = 0,1, 60% das classes ficam concentradas e o ganho é negativo (seção 2); na "
            "seed 44 com α = 0,05, 50% e o ganho é dos menores (+6,5), porque os dois "
            "modelos vão mal (39% e 33%).",
        ]),
    ]

    # 5 ----------------------------------------------------------------
    s += [
        H1("5. Conclusões, limitações e próximos passos"),
        H2("O que já se pode afirmar"),
        bullets([
            "Com comunicação realista, o LeanFed original não é melhor que o FedAvg. A "
            "correção que reserva comunicação e custo fixo é o que faz a política "
            "funcionar.",
            "O LeanFed-c traz ganho quando a heterogeneidade é de rótulo e as classes se "
            "concentram em clientes com bateria limitada. Com 8 seeds no CIFAR-10, o ganho "
            "é claro com α = 0,05 (+9,1 pontos, 8 de 8 seeds) e provável com α = 0,1 (+4,3, "
            "6 de 8, p = 0,055). No FEMNIST com Dirichlet, +1,6 com 3 seeds. Sem essa "
            "concentração, ele empata com o FedAvg ou fica até 0,4 ponto abaixo.",
            "O ganho do LeanFed-c vem de manter os clientes de bateria pequena, e as "
            "classes deles, até o fim do treino; o pico de acurácia não melhora.",
            "A fração mínima 0,5 melhora o LeanFed-c nos quatro α do CIFAR-10 (+7,3 a +0,5 "
            "pontos) e o deixa à frente do FedAvg em α = 0,05, 0,1 e 0,3 (+16,3, +6,8 e "
            "+1,6), empatado em α = 1,0, com pico maior ou igual em todos. É a melhor "
            "política medida em todos os α.",
            "O ganho da fração mínima vem de pular rodadas, e não de treinar uma fração "
            "dos dados: os clientes de bateria pequena guardam energia no começo e treinam "
            "todos os dados no fim. No α = 0,1, \"treina tudo ou pula\" (fração 1,0) é a "
            "melhor variante (+4,1 sobre o LeanFed-c, +8,5 sobre o FedAvg).",
            "Para comparar políticas, usar a média das últimas rodadas e várias seeds; a "
            "acurácia de uma rodada e de uma seed é ruído. Três seeds não bastam: no "
            "α = 0,1, as seeds 42 a 44 deram o dobro do ganho do LeanFed-c medido com 8, e "
            "o ganho com D = 5 caiu de 10 para 6,3 pontos. Já a fração mínima, que com 3 "
            "seeds parecia ruído, se confirmou com 8.",
        ]),
        H2("Limitações"),
        bullets([
            "O FEMNIST tem só 3 seeds (42 a 44) e nenhuma variante do LeanFed-c. A fração "
            "mínima 0,5 tem os quatro α do CIFAR-10; as frações 0,3, 0,7 e 1,0, a margem de "
            "cauda e D = 5 só o α = 0,1.",
            "A métrica é a média das 3 últimas rodadas, e a fração mínima ganha justamente "
            "concentrando os clientes frágeis no fim. Ainda não foi medido se o modelo "
            "dessas rodadas também é melhor em métricas que não dependem do momento de "
            "parada, como a acurácia por classe ou o pior caso entre classes.",
            "A fração de classes concentradas explica a diferença entre cenários, mas não "
            "entre seeds do mesmo cenário: falta uma medida que preveja quando o FedAvg "
            "realmente degrada no fim.",
            "Setup único: uma faixa de rede (25 Mbps), sem diferença de potência entre "
            "clientes, sem recarga. A energia vem do perfil de uma GPU de desktop, não de "
            "um celular.",
            "As mortes no fim por uma rodada de rede ruim continuam sem solução: a margem "
            "de cauda troca a morte por rodadas puladas.",
            "Só existe o modelo completo: os submodelos de largura variável ainda não "
            "foram implementados.",
        ]),
        H2("Próximos passos"),
        bullets([
            "Testar a fração mínima 0,5 no FEMNIST e repetir o FEMNIST nas seeds 45 a 49.",
            "Comparar a acurácia por classe com e sem a fração mínima, para confirmar que o "
            "ganho vem das classes dos clientes de 500 J.",
            "Entender os casos em que o FedAvg perde classes e não degrada (seed 45 com "
            "α = 0,1), olhando a acurácia por classe.",
            "Reavaliar a ideia de fazer os clientes frágeis contribuírem mais cedo: a "
            "fração mínima ganha fazendo o contrário.",
            "Variar o setup (faixas de rede, baterias, potência por cliente) para ver se a "
            "vantagem se mantém.",
            "Submodelos de largura variável (HeteroFL), para comparar \"reduzir dados\" "
            "(LeanFed-c) com \"reduzir modelo\".",
            "Para decidir: um terceiro dataset como o EuroSAT só faz sentido se a "
            "dissertação mirar sensoriamento remoto, porque repete a condição do CIFAR-10 "
            "(10 classes, sem partição natural).",
        ]),
    ]

    # A ----------------------------------------------------------------
    s += [
        H1("Apêndice A. Runs e como reproduzir"),
        P("Pastas em <i>outputs/</i> (prefixo do id). Todas com o cenário da seção 1; "
          "cada linha muda só o que o nome diz. As demais runs estão no apêndice do "
          "histórico."),
        table(
            ["Cenário e política", "Seed 42", "Seed 43", "Seed 44"],
            [
                ["CIFAR-10 α = 0,05: FedAvg", "17512…", "15739…", "68363…"],
                ["CIFAR-10 α = 0,05: LeanFed-c", "13324…", "56224…", "74911…"],
                ["CIFAR-10 α = 0,05: fração mínima 0,5", "74231…", "39955…", "21686…"],
                ["CIFAR-10 α = 0,1: FedAvg", "1977…", "18006…", "10759…"],
                ["CIFAR-10 α = 0,1: LeanFed", "16069…", "11656…", "11938…"],
                ["CIFAR-10 α = 0,1: LeanFed-c", "10494…", "12096…", "4501…"],
                ["CIFAR-10 α = 0,1: fração mínima 0,3", "70223…", "12763…", "12243…"],
                ["CIFAR-10 α = 0,1: fração mínima 0,5", "57608…", "11724…", "21629…"],
                ["CIFAR-10 α = 0,1: fração mínima 0,7", "94830…", "11212…", "36821…"],
                ["CIFAR-10 α = 0,1: fração mínima 1,0", "39105…", "11630…", "13244…"],
                ["CIFAR-10 α = 0,1: margem de cauda 0,9", "17135…", "16908…", "11427…"],
                ["CIFAR-10 α = 0,1, D = 5: FedAvg", "99297…", "14583…", "15387…"],
                ["CIFAR-10 α = 0,1, D = 5: LeanFed", "41686…", "16970…", "12657…"],
                ["CIFAR-10 α = 0,1, D = 5: LeanFed-c", "10067…", "10184…", "98610…"],
                ["CIFAR-10 α = 0,3: FedAvg", "11428…", "40976…", "15233…"],
                ["CIFAR-10 α = 0,3: LeanFed-c", "14875…", "52897…", "68942…"],
                ["CIFAR-10 α = 0,3: fração mínima 0,5", "10723…", "10638…", "15043…"],
                ["CIFAR-10 α = 1,0: FedAvg", "12018…", "21996…", "11654…"],
                ["CIFAR-10 α = 1,0: LeanFed-c", "17183…", "69488…", "15431…"],
                ["CIFAR-10 α = 1,0: fração mínima 0,5", "18343…", "49975…", "67601…"],
                ["FEMNIST escritor: FedAvg", "10788…", "31018…", "31394…"],
                ["FEMNIST escritor: LeanFed-c", "17288…", "11376…", "72789…"],
                ["FEMNIST Dirichlet: FedAvg", "38162…", "12225…", "15641…"],
                ["FEMNIST Dirichlet: LeanFed-c", "14124…", "14952…", "15300…"],
            ],
            [2.4, 1.2, 1.2, 1.2],
            split=True,
        ),
        table(
            ["Cenário e política", "Seed 45", "Seed 46", "Seed 47", "Seed 48", "Seed 49"],
            [
                ["CIFAR-10 α = 0,05: FedAvg", "11973…", "10618…", "13286…", "18104…", "75581…"],
                ["CIFAR-10 α = 0,05: LeanFed-c", "17529…", "19458…", "80657…", "10299…",
                 "40953…"],
                ["CIFAR-10 α = 0,05: fração mínima 0,5", "94595…", "16242…", "15379…",
                 "33992…", "11574…"],
                ["CIFAR-10 α = 0,1: FedAvg", "84285…", "72545…", "11917…", "11541…", "13836…"],
                ["CIFAR-10 α = 0,1: LeanFed", "12960…", "14051…", "63580…", "13382…", "92632…"],
                ["CIFAR-10 α = 0,1: LeanFed-c", "12443…", "15957…", "22576…", "91925…",
                 "40049…"],
                ["CIFAR-10 α = 0,1: fração mínima 0,3", "58980…", "14473…", "90559…",
                 "83106…", "97801…"],
                ["CIFAR-10 α = 0,1: fração mínima 0,5", "17593…", "38953…", "74931…",
                 "13337…", "13428…"],
                ["CIFAR-10 α = 0,1: fração mínima 0,7", "10531…", "68311…", "13025…",
                 "14309…", "92826…"],
                ["CIFAR-10 α = 0,1: fração mínima 1,0", "79835…", "14346…", "13550…",
                 "17814…", "57671…"],
                ["CIFAR-10 α = 0,1: margem de cauda 0,9", "80290…", "14219…", "18360…",
                 "15014…", "24191…"],
                ["CIFAR-10 α = 0,1, D = 5: FedAvg", "16348…", "58377…", "55821…", "18038…",
                 "63194…"],
                ["CIFAR-10 α = 0,1, D = 5: LeanFed", "43062…", "39301…", "17083…", "13518…",
                 "13321…"],
                ["CIFAR-10 α = 0,1, D = 5: LeanFed-c", "55191…", "12880…", "10570…", "10191…",
                 "16714…"],
                ["CIFAR-10 α = 0,3: FedAvg", "48566…", "14967…", "14939…", "15273…", "15319…"],
                ["CIFAR-10 α = 0,3: LeanFed-c", "48229…", "10455…", "13603…", "14944…",
                 "10437…"],
                ["CIFAR-10 α = 0,3: fração mínima 0,5", "16108…", "15101…", "15890…",
                 "21442…", "17025…"],
                ["CIFAR-10 α = 1,0: FedAvg", "60910…", "13293…", "12234…", "84912…", "14615…"],
                ["CIFAR-10 α = 1,0: LeanFed-c", "20015…", "13867…", "16467…", "37089…",
                 "75707…"],
                ["CIFAR-10 α = 1,0: fração mínima 0,5", "88964…", "27903…", "58342…",
                 "18979…", "15558…"],
            ],
            [2.4, 1, 1, 1, 1, 1],
            split=True,
        ),
        P("Com os containers de pé (<i>./run.sh -n 15 -b 1500,1000,500 --net-mbps 25</i>):"),
        code("""
# lotes (um por cenário)
./scripts/run_seeds.sh -v fedavg,leanfed,leanfed-c 42 43 44           # CIFAR-10, α = 0,1
./scripts/run_seeds.sh -v fedavg,leanfed,leanfed-c 45 46 47 48 49     # seeds extras
./scripts/run_seeds.sh -v leanfed-c-mdf0.3,leanfed-c-mdf0.5,leanfed-c-tail0.9 42 43 44 45 46 47 48 49
./scripts/run_seeds.sh -x "energy-fixed-j=5.0" -v fedavg,leanfed,leanfed-c 42 43 44 45 46 47 48 49
./scripts/run_seeds.sh -v leanfed-c-mdf0.7,leanfed-c-mdf1.0 42 43 44 45 46 47 48 49
./scripts/run_seeds.sh -d cifar10-a0.3 -v fedavg,leanfed-c 42 43 44   # ou a0.05, a1.0
./scripts/run_seeds.sh -d cifar10-a0.3 -v leanfed-c-mdf0.5 42 43 44 45 46 47 48 49
./scripts/run_seeds.sh -d femnist -v fedavg,leanfed,leanfed-c 42 43 44
./scripts/run_seeds.sh -d femnist-dirichlet -v fedavg,leanfed,leanfed-c 42 43 44
# média ± desvio, classes perdidas e ganho pareado (o --ref escolhe o cenário)
python3 scripts/seeds_summary.py --ref 1977  -o graficos/seeds         # α = 0,1 e variantes
python3 scripts/seeds_summary.py --ref 11428 -o graficos/seeds-alfa-0.3
python3 scripts/class_loss.py    --ref 11428 -o graficos/seeds-alfa-0.3
# Figura 3 a partir dos perda_classes.csv
python3 scripts/plot_gain_classes.py --ext pdf
"""),
    ]
    return s


def main() -> None:
    figure_accuracy(TMP / "acuracia.png")
    figure_alpha(TMP / "alfa.png")
    figure_gain_classes(TMP / "ganho_classes.png")
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.8 * inch, bottomMargin=0.9 * inch,
        title="Resultados do AprendFed",
        subject="LeanFed-c contra FedAvg com bateria limitada, até 23/10/2026",
        author="AprendFed",
    )
    doc.build(build(), onFirstPage=footer, onLaterPages=footer)
    print(f"{OUT} gerado.")


if __name__ == "__main__":
    main()
