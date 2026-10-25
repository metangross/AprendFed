#!/usr/bin/env python3
"""Gera docs/resultados.pdf (resultados consolidados do AprendFed até 24/10/2026).

Uso:
    python3 docs/fontes/resultados.py [saida.pdf]

Organizado por achado; a ordem dos experimentos e o porquê de cada decisão estão
no historico_projeto.pdf. Requer reportlab, matplotlib e as fontes DejaVu. Os
números vêm de graficos/seeds*/runs_seeds.csv e perda_classes.csv
(scripts/seeds_summary.py e scripts/class_loss.py, com --bn no FEMNIST); estão
escritos aqui para o PDF não depender de outputs/, que não é versionado.
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
    canvas.drawString(MARGIN, y, "Resultados do projeto AprendFed — até 25 de outubro de 2026")
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
# contribuem em nenhuma das 3 últimas rodadas). Seeds 42 a 49, nessa ordem. No
# FEMNIST, a acurácia é a com a BatchNorm recalibrada (test_acc_bn).
SETUPS = {
    # nome: (dataset, rótulo curto, α no eixo da Figura 3 ou None, FedAvg, LeanFed-c, classes)
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
                                    (0.7022, 0.7654, 0.7554, 0.7809, 0.7206, 0.7497,
                                     0.7583, 0.7469),
                                    (0.7349, 0.7715, 0.7575, 0.7744, 0.7473, 0.7409,
                                     0.7851, 0.7723),
                                    (0.2258, 0.1935, 0.1774, 0.1774, 0.0968, 0.0968,
                                     0.1290, 0.1452)),
    "FEMNIST, por escritor": ("FEMNIST", "por escritor", None,
                              (0.8400, 0.8446, 0.8454, 0.8451, 0.8437, 0.8446, 0.8440, 0.8502),
                              (0.8389, 0.8451, 0.8434, 0.8417, 0.8407, 0.8455, 0.8427, 0.8469),
                              (0.0,) * 8),
}
# LeanFed-c com fração mínima 0,5, acurácia média das 3 últimas rodadas, seeds 42 a 49.
MDF05 = {
    "CIFAR-10, α = 0,05": (0.7229, 0.6978, 0.5890, 0.6598, 0.7706, 0.5718, 0.5279, 0.6062),
    "CIFAR-10, α = 0,1": (0.7915, 0.7251, 0.7545, 0.7977, 0.8078, 0.7623, 0.7780, 0.7653),
    "CIFAR-10, α = 0,3": (0.8646, 0.8534, 0.8671, 0.8486, 0.8620, 0.8578, 0.8581, 0.8611),
    "CIFAR-10, α = 1,0": (0.8790, 0.8790, 0.8744, 0.8713, 0.8858, 0.8733, 0.8768, 0.8796),
    "FEMNIST, Dirichlet α = 0,1": (0.7776, 0.7877, 0.8020, 0.7990, 0.7988, 0.7541, 0.7980,
                                   0.7823),
    "FEMNIST, por escritor": (0.8421, 0.8487, 0.8487, 0.8447, 0.8454, 0.8478, 0.8467, 0.8511),
}
# Causa do ganho da fração mínima (CIFAR-10, α = 0,1): acurácia média das 3 últimas
# rodadas, seeds 42 a 49, com rádio (1,8 W), sem rádio (--radio-w 0, baterias 1025, 550
# e 85 J) e com margem de cauda 0,9.
CAUSE = {
    ("rádio", "LeanFed-c"): (0.8043, 0.6938, 0.7134, 0.7545, 0.7810, 0.7279, 0.7516, 0.7539),
    ("rádio", "fração 0,5"): MDF05["CIFAR-10, α = 0,1"],
    ("rádio", "fração 1,0"): (0.8125, 0.7615, 0.7260, 0.8229, 0.7993, 0.8056, 0.7958, 0.7885),
    ("sem rádio", "LeanFed-c"): (0.8004, 0.7468, 0.7447, 0.7886, 0.7487, 0.7504, 0.7454,
                                 0.7562),
    ("sem rádio", "fração 0,5"): (0.8034, 0.7537, 0.7331, 0.8155, 0.7526, 0.7890, 0.7525,
                                  0.7742),
    ("sem rádio", "fração 1,0"): (0.7987, 0.7456, 0.7022, 0.7819, 0.7641, 0.7857, 0.7609,
                                  0.7905),
    ("cauda", "LeanFed-c"): (0.8036, 0.7091, 0.7100, 0.7968, 0.7700, 0.7311, 0.7432, 0.7659),
    ("cauda", "fração 0,5"): (0.8002, 0.7567, 0.7425, 0.8188, 0.7969, 0.7833, 0.7749,
                              0.7921),
    ("cauda", "fração 1,0"): (0.8122, 0.7687, 0.7021, 0.8260, 0.7930, 0.7987, 0.7951, 0.7986),
}
# Médias das 8 seeds, dos logs dos clientes: exemplos treinados em 20 rodadas e por
# rodada nas rodadas 18 a 20 (mil), energia de rede e de treino (kJ), participações,
# clientes sem bateria no fim.
CAUSE_LOGS = {
    ("rádio", "LeanFed-c"): (505, 25.4, 6.71, 4.51, 285, 1.6),
    ("rádio", "fração 0,5"): (563, 31.2, 6.15, 5.04, 260, 1.2),
    ("rádio", "fração 1,0"): (583, 38.5, 5.66, 5.18, 240, 0.1),
    ("sem rádio", "LeanFed-c"): (529, 26.4, 0.0, 4.70, 300, 0.0),
    ("sem rádio", "fração 0,5"): (529, 31.1, 0.0, 4.70, 253, 0.0),
    ("sem rádio", "fração 1,0"): (516, 36.1, 0.0, 4.58, 220, 0.0),
    ("cauda", "LeanFed-c"): (492, 27.7, 6.54, 4.38, 278, 0.2),
    ("cauda", "fração 0,5"): (548, 33.5, 5.97, 4.87, 255, 0.0),
    ("cauda", "fração 1,0"): (559, 38.4, 5.42, 4.97, 231, 0.0),
}
# Varredura da potência do rádio (CIFAR-10, α = 0,1, baterias 1500, 1000 e 500 J): acurácia
# média das 3 últimas rodadas, seeds 42 a 49. O 1,8 W é o cenário da seção 1.
RADIO = {
    0.9: ((0.7723, 0.7494, 0.7504, 0.7937, 0.8019, 0.7857, 0.7786, 0.7931),
          (0.7822, 0.7572, 0.7579, 0.8041, 0.8074, 0.7928, 0.7789, 0.7755)),
    1.8: (CAUSE[("rádio", "LeanFed-c")], CAUSE[("rádio", "fração 0,5")]),
    2.7: ((0.7775, 0.6842, 0.6691, 0.7221, 0.7067, 0.7221, 0.7127, 0.7161),
          (0.7880, 0.7259, 0.7267, 0.7933, 0.7558, 0.7389, 0.7691, 0.7232)),
}
# Como CAUSE_LOGS, por potência: (LeanFed-c, fração 0,5).
RADIO_LOGS = {
    0.9: ((644, 31.1, 3.50, 5.74, 298, 1.0), (656, 33.5, 3.39, 5.85, 288, 0.9)),
    1.8: (CAUSE_LOGS[("rádio", "LeanFed-c")], CAUSE_LOGS[("rádio", "fração 0,5")]),
    2.7: ((403, 22.8, 8.91, 3.62, 252, 2.6), (472, 29.8, 8.18, 4.25, 232, 2.0)),
}
# Fração média dos exemplos de treino nos clientes que o FedAvg perde.
DATA_LOST = {"CIFAR-10, α = 0,05": 0.578, "CIFAR-10, α = 0,1": 0.548,
             "CIFAR-10, α = 0,3": 0.450, "CIFAR-10, α = 1,0": 0.406,
             "FEMNIST, Dirichlet α = 0,1": 0.420, "FEMNIST, por escritor": 0.329}
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


def figure_radio(path: Path) -> None:
    """Ganho da fração mínima 0,5 sobre o LeanFed-c contra a potência do rádio."""
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    color = SERIES[3]
    ax.axhline(0, color=AXIS, lw=1, zorder=1)
    # Sem rádio, vazado: as baterias são outras (1025, 550 e 85 J).
    points = [(0.0, CAUSE[("sem rádio", "fração 0,5")], CAUSE[("sem rádio", "LeanFed-c")],
               CAUSE_LOGS[("sem rádio", "fração 0,5")][0]
               - CAUSE_LOGS[("sem rádio", "LeanFed-c")][0], True)]
    points += [(w, mdf, lc, RADIO_LOGS[w][1][0] - RADIO_LOGS[w][0][0], False)
               for w, (lc, mdf) in RADIO.items()]
    means = []
    for w, mdf, lc, extra, hollow in points:
        d = [100 * (x - y) for x, y in zip(mdf, lc)]
        m = statistics.mean(d)
        half = T975[len(d)] * statistics.stdev(d) / len(d) ** 0.5
        face = "none" if hollow else color
        edge = color if hollow else "white"
        ax.scatter([w] * len(d), d, s=22, facecolors=face, edgecolors=edge, alpha=0.45,
                   linewidths=1.1, zorder=2)
        ax.plot([w, w], [m - half, m + half], color=color, lw=2, zorder=3)
        ax.scatter([w], [m], s=75, facecolors="white" if hollow else color, edgecolors=color
                   if hollow else "white", linewidths=1.8, zorder=4)
        ax.annotate(f"{signed(m)}", (w, m), xytext=(9, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center", fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.text(w, -3.3, f"{signed(extra, 0)} mil exemplos", fontsize=7.5, color=INK_2,
                ha="center")
        if not hollow:
            means.append((w, m))
    ax.plot(*zip(*means), color=color, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.0, 7.7, "baterias menores", fontsize=7.5, color=MUTED, ha="center")
    ax.set_xlim(-0.35, 3.05)
    ax.set_ylim(-4, 8.5)
    ax.set_xticks([0, 0.9, 1.8, 2.7])
    ax.set_xticklabels(["0 W", "0,9 W", "1,8 W", "2,7 W"])
    ax.set_yticks(range(-2, 9, 2))
    ax.set_yticklabels([signed(v, 0) if v else "0" for v in range(-2, 9, 2)])
    ax.set_xlabel("potência do rádio", fontsize=8, color=MUTED)
    ax.set_ylabel("fração 0,5 − LeanFed-c (pontos)", fontsize=8, color=MUTED)
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


def mdf_rows(dataset):
    rows = []
    for name, mdf in MDF05.items():
        ds, label, _, fedavg, leanfed_c, _ = SETUPS[name]
        if ds != dataset:
            continue
        rows.append([label.removeprefix("α = "), mean_sd(fedavg, "%"), mean_sd(leanfed_c, "%"),
                     mean_sd(mdf, "%"), paired(mdf, leanfed_c), paired(mdf, fedavg)])
    return rows


def cause_rows():
    rows = []
    for setup, label in [("rádio", "Com rádio"), ("sem rádio", "Sem rádio"),
                         ("cauda", "Margem de cauda")]:
        for variant in ("LeanFed-c", "fração 0,5", "fração 1,0"):
            total, late, _, _, _, dead = CAUSE_LOGS[(setup, variant)]
            acc = CAUSE[(setup, variant)]
            gain = ("—" if variant == "LeanFed-c"
                    else paired(acc, CAUSE[(setup, "LeanFed-c")]))
            rows.append([label if variant == "LeanFed-c" else "", variant,
                         f"{br(total, 0)} mil", f"{br(late)} mil", br(dead), mean_sd(acc),
                         gain])
    return rows


def radio_rows():
    rows = []
    for w, (lc, mdf) in RADIO.items():
        (lc_total, _, lc_net, _, _, lc_dead), (total, _, net, _, _, dead) = RADIO_LOGS[w]
        rows.append([f"{br(w)} W", mean_sd(lc), mean_sd(mdf),
                     f"{br(lc_total, 0)} → {br(total, 0)} mil",
                     f"{br(lc_net, 2)} → {br(net, 2)}", f"{br(lc_dead)} → {br(dead)}",
                     paired(mdf, lc)])
    return rows


def build() -> list:
    s = []
    s += [
        Spacer(1, 30),
        P("Resultados do AprendFed", "title"),
        P("LeanFed-c contra FedAvg com bateria limitada: o que os experimentos mostram",
          "sub"),
        P("Até 25 de outubro de 2026 — CIFAR-10 e FEMNIST com 8 seeds — branch "
          "<i>leanfed-baseline</i> (commit 86ab3a9)",
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
             "No FEMNIST, com a acurácia medida depois de recalibrar a BatchNorm (seção 1), "
             "ela também é a melhor: +4,0 pontos sobre o FedAvg com Dirichlet α = 0,1 (8 de "
             "8 seeds) e +0,2 na partição por escritor, onde os três métodos ficam a menos "
             "de meio ponto. "
             "Com ela, os clientes de bateria pequena pulam o começo do treino e chegam ao "
             "fim com energia para treinar todos os dados; quanto maior a fração mínima, "
             "maior o ganho (com 1,0, +8,5 sobre o FedAvg em α = 0,1). Esse ganho extra se "
             "espalha por todas as classes, e não só pelas desses clientes, e sobe "
             "sobretudo a pior classe. Ele vem de treinar mais exemplos no total: cada "
             "rodada pulada economiza a energia de rede, que vira treino. Sem rádio, o "
             "total fica igual e o ganho cai para +1,2 ponto (fração 0,5, sem "
             "significância); com a margem de cauda, ninguém morre e o ganho fica (+2,9). "
             "Variando a potência do rádio, o ganho acompanha o custo da rede: +0,4 com "
             "0,9 W, +2,5 com 1,8 W e +3,9 com 2,7 W (8 de 8 seeds). "
             "Entre "
             "cenários, o ganho acompanha a fração de classes concentradas nos clientes que "
             "o FedAvg perde por falta de bateria, e não a quantidade de dados que esses "
             "clientes levam; seed a seed, essa fração não prevê o ganho."),
        P("<b>Conteúdo</b>"),
        bullets([
            "1. Cenário, políticas e métricas",
            "2. O LeanFed-c ganha porque não perde clientes no fim",
            "3. Ajustes do LeanFed-c: a fração mínima ajuda em todos os cenários",
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
            "estão vivos. No FEMNIST, medida com a BatchNorm recalibrada (abaixo).",
            "<b>Ganho</b>: acurácia do LeanFed-c menos a do FedAvg <b>com a mesma seed</b> "
            "(mesma partição e mesma sequência de vazões), com intervalo de confiança de "
            "95% pela t de Student sobre as diferenças pareadas.",
            "<b>Seeds</b>: valores são média ± desvio amostral sobre as seeds 42 a 49 "
            "(n = 8), em todos os cenários. Diferenças entre duas variantes do LeanFed-c "
            "também são pareadas por seed.",
            "<b>Cliente perdido</b>: não contribuiu em nenhuma das 3 últimas rodadas. "
            "<b>Classes concentradas</b>: fração das classes com mais de 80% dos seus "
            "exemplos de treino nos clientes perdidos pelo FedAvg.",
        ]),
        H2("Avaliação no FEMNIST: BatchNorm recalibrada"),
        P("A FedAvg tira a média das estatísticas das camadas BatchNorm (média e variância "
          "das ativações) junto com a dos pesos. No FEMNIST não-IID, essa média não "
          "representa os dados, e a acurácia do modelo global no teste fica subestimada e "
          "às vezes desaba numa rodada, com todos os clientes treinando normalmente: na "
          "rodada 1, 9% contra 33% recalibrado (FedAvg com Dirichlet, média das seeds); "
          "das rodadas 5 a 20, 32 das 768 avaliações caem abaixo de 30%, a pior a 2,4%."),
        P("Por isso, nas runs do FEMNIST o servidor também avalia o modelo depois de refazer "
          "essas estatísticas com 2048 exemplos de treino dos clientes, sorteados pela seed "
          "(sem usar o teste). Os pesos não mudam, e o modelo enviado aos clientes também "
          "não: a recalibração só entra na medida. Com ela, a pior avaliação das rodadas 5 "
          "a 20 fica em 57,6%, e a média das 3 últimas rodadas sobe de 5 a 16 pontos, "
          "conforme o método. Todas as acurácias do FEMNIST neste documento são as "
          "recalibradas."),
        note("<b>Sem recalibrar, o FEMNIST não compara métodos.</b> O desvio entre seeds "
             "chega a 18 pontos (contra 2 recalibrado), e a ordem muda: com Dirichlet, a "
             "fração mínima 0,5 ficaria em último (62,7% contra 69,5% do FedAvg); por "
             "escritor, em primeiro (78,8% contra 74,6%). No CIFAR-10, as quedas de acurácia "
             "são reais (vêm dos clientes que saem) e recalibrar não as recupera; ele usa a "
             "avaliação original."),
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
        H1("3. Ajustes do LeanFed-c: a fração mínima ajuda em todos os cenários"),
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
        H2("O que a fração mínima muda na participação"),
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
            "quase todos os dados nas últimas rodadas. Quanto maior o mínimo, mais a "
            "participação se concentra no fim, na mesma ordem do ganho da tabela acima.",
            "É o oposto de fazer os clientes frágeis contribuírem mais cedo: com a fração "
            "mínima, eles contribuem <b>tarde</b>, e com mais dados.",
        ]),
        H2("Acurácia por classe: o ganho não fica nas classes desses clientes"),
        P("Para ver se o ganho vem das classes dos clientes de 500 J, o modelo final de "
          "cada run foi avaliado por classe no teste do CIFAR-10 (α = 0,1, 8 seeds). "
          "\"Classes dos clientes de 500 J\" são as que têm mais da metade dos exemplos de "
          "treino nesses clientes: de 2 a 4 por seed, 29% do total. Só existe o modelo da "
          "rodada 20, a da queda em todas as curvas, por isso a acurácia global é um pouco "
          "menor que a média das 3 últimas rodadas."),
        table(
            ["α = 0,1, rodada 20", "Global", "Classes dos clientes de 500 J",
             "Outras classes", "Pior classe"],
            [
                ["FedAvg", "68,8%", "68,0%", "68,7%", "21,2%"],
                ["LeanFed-c", "73,2%", "75,1%", "72,4%", "32,8%"],
                ["fração mínima 0,5", "75,2%", "77,1%", "74,2%", "39,6%"],
                ["fração mínima 0,7", "76,0%", "81,3%", "73,6%", "40,0%"],
                ["fração mínima 1,0", "78,6%", "80,1%", "77,3%", "46,2%"],
            ],
            [1.6, 0.9, 1.5, 1.1, 1.0],
        ),
        bullets([
            "<b>O ganho sobre o LeanFed-c se espalha.</b> A fração mínima 0,5 sobe +2,0 "
            "pontos nas classes dos clientes de 500 J e +1,9 nas outras. Essas classes "
            "levam 37% do ganho, quase a parte delas no total (29%), e a correlação entre o "
            "ganho de cada classe e a parte dela nesses clientes é fraca (Spearman +0,07 em "
            "média por seed; +0,15 com a fração 1,0), sem significância. Com α = 0,05 há "
            "alguma concentração (62% do ganho em 31% das classes), com intervalo largo.",
            "<b>A pior classe é a que mais sobe.</b> Com a fração 1,0, +13,5 pontos sobre o "
            "LeanFed-c (IC 95% de +5,3 a +21,6; 7 de 8 seeds) e +25,0 sobre o FedAvg (+8,5 a "
            "+41,5), contra +5,4 e +9,8 na acurácia global. Com a fração 0,5, +6,8 sobre o "
            "LeanFed-c, sem significância.",
            "<b>Então o ganho não vem das classes desses clientes.</b> A participação "
            "deles muda como a tabela anterior mostra, mas o ganho se espalha. A "
            "subseção seguinte mostra de onde ele vem.",
        ]),
        H2("De onde vem o ganho: mais exemplos treinados no total"),
        P("Havia duas hipóteses. (1) As atualizações treinadas com poucos dados do "
          "LeanFed-c atrapalham o modelo global. Ela é fraca desde o início: a agregação "
          "já pesa cada cliente pelos exemplos que ele treinou na rodada. (2) A fração "
          "mínima faz entrar mais exemplos no treino. Os logs dos clientes mostram que "
          "entram: no α = 0,1, de 505 mil exemplos treinados em 20 rodadas no LeanFed-c "
          "para 563 mil com a fração 0,5 e 583 mil com a 1,0, na mesma ordem do ganho."),
        P("Isso acontece porque cada participação paga a rede (cerca de 23,5 J para baixar "
          "e enviar o modelo), seja qual for a fração dos dados treinada. Quando o cliente "
          "pula uma rodada, ele não gasta essa energia e pode usá-la para treinar. Com a "
          "fração 1,0, a energia de rede cai de 6,7 para 5,7 kJ e a de treino sobe de "
          "4,5 para 5,2 kJ. Dois experimentos isolam esse efeito (α = 0,1, 8 seeds):"),
        bullets([
            "<b>Sem rádio</b> (<i>--radio-w 0</i>): a rede não custa nada, e pular uma "
            "rodada deixa de economizar energia. As baterias caem para 1025, 550 e 85 J, "
            "para a energia de treino ficar parecida com a do caso com rádio.",
            "<b>Com margem de cauda 0,9</b> nas três políticas: ninguém fica sem bateria, "
            "e a diferença na morte de clientes entre o LeanFed-c (1,6 clientes) e a "
            "fração mínima sai da comparação.",
        ]),
        table(
            ["α = 0,1", "Variante", "Exemplos em 20 rodadas", "Por rodada, 18 a 20",
             "Sem bateria no fim", "Média das 3 últimas",
             "Diferença para o LeanFed-c [IC 95%], seeds melhores"],
            cause_rows(),
            [0.85, 1.0, 0.95, 0.85, 0.8, 1.12, 1.95],
        ),
        bullets([
            "<b>Sem rádio, o total de exemplos fica igual e o ganho quase some</b>: os "
            "três treinam cerca de 520 mil exemplos, e a fração mínima fica em +1,2 (0,5; "
            "p = 0,08) e +0,6 ponto (1,0; p = 0,5), contra +2,5 e +4,1 com rádio. A "
            "participação continua concentrada no fim (36 mil exemplos por rodada nas "
            "últimas rodadas com a fração 1,0, contra 26 mil no LeanFed-c), e a fração 1,0 "
            "é a que mais perde. Concentrar os dados no fim, sozinho, explica pouco.",
            "<b>Com margem de cauda, o ganho fica</b>: +2,9 (0,5; p = 0,002) e +3,3 pontos "
            "(1,0; p = 0,008) sobre o LeanFed-c com a mesma margem, sem ninguém sem bateria "
            "nas variantes com fração mínima. A diferença de exemplos também fica "
            "(492 mil contra 548 e 559 mil). A morte de clientes no fim do LeanFed-c não "
            "explica o ganho.",
            "<b>O ganho vem de treinar mais exemplos no total</b>, com a energia que a "
            "fração mínima deixa de gastar com a rede. Por isso ele não fica nas classes "
            "dos clientes de 500 J: o que muda é a quantidade de dados que entra no treino "
            "inteiro. Com a margem de cauda, a fração 0,5 chega a 78,3% (+1,0 ponto sobre "
            "ela sem margem, sem significância); a 1,0 fica igual (78,7%).",
            "Seed a seed, a relação é mais fraca. Com a fração 1,0, a seed que ganha mais "
            "exemplos sobre o LeanFed-c tende a ganhar mais acurácia (Spearman +0,71, "
            "p = 0,05); com a 0,5, não (+0,12). Com 8 seeds, uma diferença de ~60 mil "
            "exemplos (cerca de 10%) é pequena diante da variação entre seeds.",
        ]),
        P("<b>Se a causa é a rede, o ganho deve crescer com o custo dela.</b> Para testar, "
          "a potência do rádio variou de 1,8 W para 0,9 e 2,7 W, com as mesmas baterias "
          "(1500, 1000 e 500 J), no LeanFed-c e na fração 0,5 (α = 0,1, 8 seeds)."),
        table(
            ["Rádio", "LeanFed-c", "Fração 0,5", "Exemplos em 20 rodadas", "Rede (kJ)",
             "Sem bateria no fim", "Fração 0,5 − LeanFed-c [IC 95%], seeds melhores"],
            radio_rows(),
            [0.65, 1.12, 1.12, 1.2, 1.05, 0.85, 1.55],
        ),
        figure(TMP / "radio.png", 6.3,
               "Figura 2. Ganho da fração mínima 0,5 sobre o LeanFed-c contra a potência "
               "do rádio (CIFAR-10, α = 0,1). Cada ponto pequeno é uma seed; o grande, a "
               "média, com o IC 95%. Embaixo, quantos exemplos a mais a fração 0,5 treina "
               "em 20 rodadas. O 0 W (vazado) usou baterias de 1025, 550 e 85 J e não entra "
               "na linha tracejada."),
        bullets([
            "<b>O ganho cresce com a potência</b>: +0,4 ponto com 0,9 W (sem "
            "significância, p = 0,27), +2,5 com 1,8 W e +3,9 com 2,7 W (8 de 8 seeds, "
            "p = 0,003). Junto cresce a folga de exemplos: +12, +58 e +69 mil.",
            "<b>Com rede barata, pular rodadas quase não economiza</b>: com 0,9 W, a rede "
            "custa 3,5 kJ no LeanFed-c e a fração 0,5 economiza só 0,1 kJ dela (0,6 kJ com "
            "1,8 W; 0,7 kJ com 2,7 W). O ganho fica perto do caso sem rádio (+1,2).",
            "<b>Com rede cara, os dois pioram, mas o LeanFed-c piora mais</b>: de 1,8 para "
            "2,7 W, ele cai de 74,8% para 71,4% e a fração 0,5 de 77,3% para 75,3%. Os "
            "clientes sem bateria no fim sobem para 2,6 e 2,0, mas, como a margem de cauda "
            "mostrou, as mortes explicam pouco do ganho.",
            "É a previsão da causa (A): o ganho da fração mínima mede quanto da bateria "
            "a rede consome, e some quando a rede é barata.",
        ]),
        H2("A fração mínima 0,5 nos outros α"),
        table(
            ["α", "FedAvg", "LeanFed-c", "Fração mínima 0,5",
             "Fração 0,5 − LeanFed-c [IC 95%], seeds melhores",
             "Fração 0,5 − FedAvg [IC 95%], seeds melhores"],
            mdf_rows("CIFAR-10"),
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
        H2("A fração mínima 0,5 no FEMNIST"),
        table(
            ["Partição", "FedAvg", "LeanFed-c", "Fração mínima 0,5",
             "Fração 0,5 − LeanFed-c [IC 95%], seeds melhores",
             "Fração 0,5 − FedAvg [IC 95%], seeds melhores"],
            mdf_rows("FEMNIST"),
            [1.0, 1.2, 1.2, 1.1, 1.6, 1.6],
        ),
        bullets([
            "<b>Com Dirichlet α = 0,1, é a melhor política</b>: +2,7 pontos sobre o "
            "LeanFed-c e +4,0 sobre o FedAvg, nas 8 seeds nos dois casos (p = 0,003 e "
            "0,004). O LeanFed-c sozinho ganha +1,3 do FedAvg, sem significância. O pico "
            "também sobe: 79,4% contra 77,6% do FedAvg e 76,7% do LeanFed-c.",
            "<b>Por escritor, os três empatam na prática</b> (84,3% a 84,7%). A fração 0,5 "
            "ainda fica à frente do FedAvg (+0,2; 7 de 8) e do LeanFed-c (+0,4; 8 de 8), e o "
            "LeanFed-c sozinho fica 0,2 ponto abaixo do FedAvg (2 de 8), como no CIFAR-10 "
            "com α = 1,0.",
            "É o que se esperava antes das runs: ganho na partição Dirichlet e empate por "
            "escritor, onde nenhuma classe fica concentrada nos clientes que o FedAvg "
            "perde (seção 4).",
        ]),
        note("<b>O que só foi medido no α = 0,1 do CIFAR-10.</b> As frações 0,3, 0,7 e 1,0, "
             "a margem de cauda, o custo fixo D = 5 e os experimentos sem rádio, com "
             "margem de cauda na fração mínima e com outras potências do rádio."),
    ]

    # 4 ----------------------------------------------------------------
    s += [
        H1("4. O ganho depende da concentração de classes"),
        P("<b>Pergunta.</b> O LeanFed-c mantém os clientes vivos em todos os cenários. "
          "Quando isso vale acurácia? Foram testados dois tipos de cenário, cada um com "
          "FedAvg e LeanFed-c (8 seeds em cada):"),
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
               "Figura 3. Acurácia média das 3 últimas rodadas contra o α do CIFAR-10, "
               "8 seeds por ponto. Os pontos claros são as seeds; no alto, o ganho médio "
               "sobre o FedAvg, na cor de cada método. Todos pioram quando o α diminui; o "
               "FedAvg piora mais. A fração mínima 0,5 (seção 3) fica acima do LeanFed-c "
               "em todos os α."),
        figure(TMP / "ganho_classes.png", 6.3,
               "Figura 4. Ganho do LeanFed-c contra a fração de classes concentradas nos "
               "clientes que o FedAvg perde. Cada ponto pequeno é uma seed (8 por "
               "cenário); o grande, a média do cenário. CIFAR-10 preenchido, FEMNIST "
               "vazado."),
        P("<b>O que os números dizem.</b>"),
        bullets([
            "<b>Perder dados não custa; perder classes custa.</b> O FedAvg perde de 32% a "
            "58% dos dados de treino em todos os cenários, mas só fica atrás quando as "
            "classes estão concentradas nos clientes que saem. Com α = 0,3, ele perde 45% "
            "dos dados e 5% das classes, e o ganho do LeanFed-c é de 0,6 ponto, sem "
            "significância (IC 95% de −0,3 a +1,5).",
            "<b>A acurácia por classe confirma.</b> No modelo da rodada 20 com α = 0,1, "
            "74% do ganho do LeanFed-c sobre o FedAvg (87% do da fração mínima 0,5) está "
            "nas classes concentradas nos clientes que o FedAvg perde, que são 32% das "
            "classes. Nelas, o FedAvg acerta 54,8%, o LeanFed-c 67,2% e a fração 0,5 "
            "72,1%; nas outras, os três ficam entre 75% e 77%. Com α = 0,05, 93% do ganho "
            "do LeanFed-c está em 39% das classes.",
            "<b>O resultado se repete no mesmo dataset.</b> A comparação entre CIFAR-10 e "
            "FEMNIST mudava o dataset junto com a partição. A varredura do α muda só a "
            "concentração e reproduz a mesma relação: +9,1, +4,3, +0,6 e −0,4 pontos de "
            "α = 0,05 a α = 1,0.",
            "<b>O FEMNIST segue a mesma relação.</b> Com Dirichlet α = 0,1, 16% das classes "
            "ficam concentradas e o ganho do LeanFed-c é de +1,3 ponto (IC 95% de −0,1 a "
            "+2,7; 6 de 8), entre os do α = 0,1 e do α = 0,3 do CIFAR-10. Por escritor, "
            "nenhuma classe fica concentrada e o LeanFed-c fica 0,2 ponto abaixo, como no "
            "α = 1,0.",
            "<b>Só α = 0,05 tem ganho firme.</b> Positivo nas 8 seeds (de +0,7 a +16,6; "
            "t pareado p = 0,002). Em α = 0,1 o intervalo encosta no zero.",
            "<b>Sem concentração, o LeanFed-c custa pouco.</b> Com α = 1,0 ele fica abaixo "
            "do FedAvg em 7 das 8 seeds (−0,4 ponto, IC 95% de −0,8 a 0,0), e a acurácia "
            "máxima também é menor (87,4% contra 88,0%). Sem classes a proteger, sobra só o "
            "custo de treinar menos dados por rodada. Com a fração mínima 0,5, esse custo "
            "some (seção 3).",
            "<b>O ganho é relativo.</b> Com α menor os dois métodos pioram (Figura 3). O "
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
            "6 de 8, p = 0,055). No FEMNIST com Dirichlet, +1,3 (6 de 8, p = 0,06). Sem essa "
            "concentração, ele empata com o FedAvg ou fica até 0,4 ponto abaixo.",
            "O ganho do LeanFed-c vem de manter os clientes de bateria pequena, e as "
            "classes deles, até o fim do treino: a acurácia por classe mostra que três "
            "quartos do ganho estão nas classes que o FedAvg perde. O pico de acurácia não "
            "melhora.",
            "A fração mínima 0,5 melhora o LeanFed-c nos quatro α do CIFAR-10 (+7,3 a +0,5 "
            "pontos) e o deixa à frente do FedAvg em α = 0,05, 0,1 e 0,3 (+16,3, +6,8 e "
            "+1,6), empatado em α = 1,0, com pico maior ou igual em todos. No FEMNIST, "
            "ganha +4,0 do FedAvg com Dirichlet (8 de 8) e +0,2 por escritor. É a melhor "
            "política medida em todos os cenários.",
            "O ganho da fração mínima vem de pular rodadas, e não de treinar uma fração "
            "dos dados: os clientes de bateria pequena guardam energia no começo e treinam "
            "todos os dados no fim. No α = 0,1, \"treina tudo ou pula\" (fração 1,0) é a "
            "melhor variante (+4,1 sobre o LeanFed-c, +8,5 sobre o FedAvg). O ganho sobre o "
            "LeanFed-c se espalha pelas classes, e a pior classe sobe mais que a média "
            "(+13,5 com a fração 1,0). Ele vem de treinar mais exemplos no total com a "
            "energia de rede que as rodadas puladas economizam: sem custo de rádio, o "
            "ganho cai para +1,2 e +0,6 ponto, sem significância; sem mortes (margem de "
            "cauda), ele fica; e ele cresce com a potência do rádio (+0,4, +2,5 e +3,9 "
            "pontos com 0,9, 1,8 e 2,7 W, na fração 0,5).",
            "Para comparar políticas, usar a média das últimas rodadas e várias seeds; a "
            "acurácia de uma rodada e de uma seed é ruído. Três seeds não bastam: no "
            "α = 0,1, as seeds 42 a 44 deram o dobro do ganho do LeanFed-c medido com 8, e "
            "o ganho com D = 5 caiu de 10 para 6,3 pontos. Já a fração mínima, que com 3 "
            "seeds parecia ruído, se confirmou com 8.",
            "Com MobileNetV3 e dados não-IID, a média das estatísticas da BatchNorm feita "
            "pela FedAvg distorce a acurácia do modelo global no FEMNIST; para comparar "
            "métodos, é preciso recalibrá-las antes de avaliar (seção 1).",
        ]),
        H2("Limitações"),
        bullets([
            "A fração mínima 0,5 tem os quatro α do CIFAR-10 e as duas partições do "
            "FEMNIST; as frações 0,3, 0,7 e 1,0, a margem de cauda e D = 5, só o α = 0,1 do "
            "CIFAR-10.",
            "No FEMNIST, a BatchNorm só é recalibrada na avaliação: os clientes continuam "
            "recebendo a média das estatísticas. Não se sabe se recalibrar também no "
            "modelo enviado (ou trocar a BatchNorm por GroupNorm) mudaria o treino, nem por "
            "que a distorção varia de 5 a 16 pontos entre os métodos.",
            "A causa do ganho da fração mínima (mais exemplos no total) só foi testada no "
            "α = 0,1 do CIFAR-10, e depende do custo do rádio: com 0,9 W o ganho some. "
            "Com uma potência só para todos os clientes, não se sabe o que acontece quando "
            "o custo da rede varia entre eles. A acurácia por classe só foi medida no "
            "modelo da rodada 20, porque os das outras rodadas não são salvos, e não se "
            "sabe se o ganho depende de parar o treino na rodada 20.",
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
            "Variar a potência do rádio entre clientes (ou a faixa de rede), em vez de uma "
            "potência só para todos, para ver se a fração mínima continua ganhando quando "
            "o custo da rede é desigual.",
            "Entender os casos em que o FedAvg perde classes e não degrada (seed 45 com "
            "α = 0,1), com a acurácia por classe (<i>scripts/class_accuracy.py</i>).",
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
          "cada linha muda só o que o nome diz. As runs do FEMNIST são as com "
          "<i>bn-recalibrate=true</i> (lote de 23/10), que repetem as anteriores com a "
          "mesma acurácia original e acrescentam a recalibrada. As demais runs estão no "
          "apêndice do histórico."),
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
                ["CIFAR-10 α = 0,1: fração 0,5 e margem de cauda 0,9", "67067…", "56917…",
                 "91968…"],
                ["CIFAR-10 α = 0,1: fração 1,0 e margem de cauda 0,9", "15522…", "18252…",
                 "10244…"],
                ["CIFAR-10 α = 0,1, sem rádio: LeanFed-c", "56265…", "11039…", "11506…"],
                ["CIFAR-10 α = 0,1, sem rádio: fração mínima 0,5", "11059…", "17811…",
                 "75048…"],
                ["CIFAR-10 α = 0,1, sem rádio: fração mínima 1,0", "53248…", "34449…",
                 "97914…"],
                ["CIFAR-10 α = 0,1, rádio 0,9 W: LeanFed-c", "28894…", "86803…", "15504…"],
                ["CIFAR-10 α = 0,1, rádio 0,9 W: fração mínima 0,5", "52212…", "12939…",
                 "23937…"],
                ["CIFAR-10 α = 0,1, rádio 2,7 W: LeanFed-c", "17904…", "79092…", "96421…"],
                ["CIFAR-10 α = 0,1, rádio 2,7 W: fração mínima 0,5", "70098…", "43013…",
                 "52032…"],
                ["CIFAR-10 α = 0,1, D = 5: FedAvg", "99297…", "14583…", "15387…"],
                ["CIFAR-10 α = 0,1, D = 5: LeanFed", "41686…", "16970…", "12657…"],
                ["CIFAR-10 α = 0,1, D = 5: LeanFed-c", "10067…", "10184…", "98610…"],
                ["CIFAR-10 α = 0,3: FedAvg", "11428…", "40976…", "15233…"],
                ["CIFAR-10 α = 0,3: LeanFed-c", "14875…", "52897…", "68942…"],
                ["CIFAR-10 α = 0,3: fração mínima 0,5", "10723…", "10638…", "15043…"],
                ["CIFAR-10 α = 1,0: FedAvg", "12018…", "21996…", "11654…"],
                ["CIFAR-10 α = 1,0: LeanFed-c", "17183…", "69488…", "15431…"],
                ["CIFAR-10 α = 1,0: fração mínima 0,5", "18343…", "49975…", "67601…"],
                ["FEMNIST escritor: FedAvg", "29553…", "14431…", "10766…"],
                ["FEMNIST escritor: LeanFed-c", "14878…", "28328…", "12001…"],
                ["FEMNIST escritor: fração mínima 0,5", "17757…", "44950…", "10449…"],
                ["FEMNIST Dirichlet: FedAvg", "12511…", "15898…", "72012…"],
                ["FEMNIST Dirichlet: LeanFed-c", "88227…", "19479…", "59824…"],
                ["FEMNIST Dirichlet: fração mínima 0,5", "45006…", "95189…", "12909…"],
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
                ["CIFAR-10 α = 0,1: fração 0,5 e margem de cauda 0,9", "58638…", "73913…",
                 "62442…", "24803…", "38960…"],
                ["CIFAR-10 α = 0,1: fração 1,0 e margem de cauda 0,9", "92410…", "15962…",
                 "15879…", "15497…", "69158…"],
                ["CIFAR-10 α = 0,1, sem rádio: LeanFed-c", "36352…", "76661…", "82841…",
                 "62932…", "99753…"],
                ["CIFAR-10 α = 0,1, sem rádio: fração mínima 0,5", "62209…", "15923…",
                 "26993…", "18326…", "94229…"],
                ["CIFAR-10 α = 0,1, sem rádio: fração mínima 1,0", "25801…", "15356…",
                 "12690…", "18217…", "17036…"],
                ["CIFAR-10 α = 0,1, rádio 0,9 W: LeanFed-c", "25716…", "59442…", "49788…",
                 "17171…", "18359…"],
                ["CIFAR-10 α = 0,1, rádio 0,9 W: fração mínima 0,5", "29361…", "35438…",
                 "24192…", "16546…", "12529…"],
                ["CIFAR-10 α = 0,1, rádio 2,7 W: LeanFed-c", "15844…", "14213…", "13321…",
                 "13060…", "83579…"],
                ["CIFAR-10 α = 0,1, rádio 2,7 W: fração mínima 0,5", "10793…", "37793…",
                 "35657…", "87928…", "12324…"],
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
                ["FEMNIST escritor: FedAvg", "14499…", "13120…", "12118…", "15009…", "14904…"],
                ["FEMNIST escritor: LeanFed-c", "12487…", "15266…", "10577…", "13214…",
                 "15799…"],
                ["FEMNIST escritor: fração mínima 0,5", "46534…", "50927…", "14157…",
                 "32677…", "11099…"],
                ["FEMNIST Dirichlet: FedAvg", "16578…", "18360…", "16197…", "44154…",
                 "81443…"],
                ["FEMNIST Dirichlet: LeanFed-c", "17665…", "15770…", "70768…", "13113…",
                 "18339…"],
                ["FEMNIST Dirichlet: fração mínima 0,5", "19907…", "77329…", "16194…",
                 "15101…", "59879…"],
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
./scripts/run_seeds.sh -x "tail-margin=0.9" -v leanfed-c-mdf0.5,leanfed-c-mdf1.0 42 43 44 45 46 47 48 49
# sem rádio: subir antes ./run.sh -n 15 -b 1025,550,85 --net-mbps 25 --radio-w 0
./scripts/run_seeds.sh -v leanfed-c,leanfed-c-mdf0.5,leanfed-c-mdf1.0 42 43 44 45 46 47 48 49
# outra potência: subir antes ./run.sh -n 15 -b 1500,1000,500 --net-mbps 25 --radio-w 0.9 (ou 2.7)
./scripts/run_seeds.sh -v leanfed-c,leanfed-c-mdf0.5 42 43 44 45 46 47 48 49
./scripts/run_seeds.sh -d cifar10-a0.3 -v fedavg,leanfed-c 42 43 44   # ou a0.05, a1.0
./scripts/run_seeds.sh -d cifar10-a0.3 -v leanfed-c-mdf0.5 42 43 44 45 46 47 48 49
./scripts/run_seeds.sh -d femnist -x "bn-recalibrate=true" \\
    -v fedavg,leanfed-c,leanfed-c-mdf0.5 42 43 44 45 46 47 48 49   # ou -d femnist-dirichlet
# média ± desvio, classes perdidas e ganho pareado (o --ref escolhe o cenário)
python3 scripts/seeds_summary.py --ref 1977  -o graficos/seeds         # α = 0,1 e variantes
python3 scripts/seeds_summary.py --ref 5626  -o graficos/seeds-sem-radio
python3 scripts/seeds_summary.py --ref 28894 -o graficos/seeds-radio-0.9    # 2,7 W: --ref 17904
python3 scripts/seeds_summary.py --ref 11428 -o graficos/seeds-alfa-0.3
python3 scripts/class_loss.py    --ref 11428 -o graficos/seeds-alfa-0.3
python3 scripts/seeds_summary.py --bn --ref 38162 -o graficos/seeds-femnist-dirichlet
python3 scripts/class_loss.py    --bn --ref 38162 -o graficos/seeds-femnist-dirichlet
python3 scripts/seeds_summary.py --bn --ref 10788 -o graficos/seeds-femnist   # escritor
# acurácia por classe do final_model.pt e parte de cada classe por bateria (na CPU)
docker run --rm -v aprendfed_data:/data:ro -v "$PWD":/app:ro aprendfed:latest \\
    python scripts/class_accuracy.py [--bn] outputs/run-<id>-<data> ...
# Figura 4 a partir dos perda_classes.csv
python3 scripts/plot_gain_classes.py --ext pdf
"""),
    ]
    return s


def main() -> None:
    figure_accuracy(TMP / "acuracia.png")
    figure_radio(TMP / "radio.png")
    figure_alpha(TMP / "alfa.png")
    figure_gain_classes(TMP / "ganho_classes.png")
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.8 * inch, bottomMargin=0.9 * inch,
        title="Resultados do AprendFed",
        subject="LeanFed-c contra FedAvg com bateria limitada, até 25/10/2026",
        author="AprendFed",
    )
    doc.build(build(), onFirstPage=footer, onLaterPages=footer)
    print(f"{OUT} gerado.")


if __name__ == "__main__":
    main()
