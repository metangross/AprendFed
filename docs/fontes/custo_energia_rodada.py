#!/usr/bin/env python3
"""Gera docs/custo_energia_rodada.pdf (notas de pesquisa sobre custo de energia).

Uso:
    python3 docs/fontes/custo_energia_rodada.py [saida.pdf]

Requer reportlab, matplotlib e as fontes DejaVu. As equações são renderizadas
com o mathtext do matplotlib e a Figura 1 é refeita a partir das contagens de
MACs e parâmetros da CNN do projeto.
"""

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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "custo_energia_rodada.pdf"
TMP = Path(tempfile.mkdtemp(prefix="custo_energia_"))

# ---------------------------------------------------------------- estilo
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
    "h3": ParagraphStyle(
        "h3", fontName="DejaVu-Bold", fontSize=10, leading=15, textColor=TEXT,
        spaceBefore=4, spaceAfter=2, keepWithNext=True,
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
    "ref": ParagraphStyle(
        "ref", fontName="DejaVu", fontSize=9.2, leading=13.5, textColor=TEXT,
        leftIndent=14, firstLineIndent=-14, spaceAfter=4,
    ),
}
S["h2free"] = ParagraphStyle("h2free", parent=S["h2"], keepWithNext=False)
S["h3free"] = ParagraphStyle("h3free", parent=S["h3"], keepWithNext=False)


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
    if len(rows) < 5 and not split:
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


def code(text, head=None):
    pre = Preformatted(text.strip("\n"), S["code"])
    t = Table([[pre]], colWidths=[WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return KeepTogether(([H2(head)] if head else []) + [t, Spacer(1, 10)])


_eq_count = 0


def eq(tex, fontsize=14):
    """Equação centralizada, renderizada com mathtext a 300 dpi."""
    global _eq_count
    _eq_count += 1
    path = TMP / f"eq{_eq_count}.png"
    fig = plt.figure()
    fig.text(0, 0, tex, fontsize=fontsize, color="#222222")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    img = Image(str(path), width=w * 72 / 300, height=h * 72 / 300)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 2), img, Spacer(1, 6)])


def footer(canvas, doc):
    canvas.saveState()
    y = 0.55 * inch
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, y + 11, A4[0] - MARGIN, y + 11)
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(MARGIN, y, "Custo de energia por rodada em FL — notas de pesquisa")
    canvas.drawRightString(A4[0] - MARGIN, y, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------- números
def cnn_cost(width: float) -> tuple[int, int]:
    """MACs (forward, 1 amostra 32x32) e parâmetros do Net de task.py com largura p."""
    c1, c2, c3, h = (max(1, int(c * width)) for c in (32, 64, 128, 256))
    convs = [(3, c1, 32), (c1, c2, 16), (c2, c3, 8)]  # (entrada, saída, lado)
    macs = sum(side * side * cout * cin * 9 for cin, cout, side in convs)
    params = sum(cin * cout * 9 + cout for cin, cout, _ in convs)
    for fin, fout in [(c3 * 16, h), (h, 10)]:
        macs += fin * fout
        params += fin * fout + fout
    return macs, params


MACS1, PARAMS1 = cnn_cost(1.0)
WIDTHS = (1.0, 0.5, 0.25, 0.125)
REL = {p: (cnn_cost(p)[0] / MACS1, cnn_cost(p)[1] / PARAMS1) for p in WIDTHS}
BITS1 = 32 * PARAMS1


def br(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def figure1(path: Path) -> None:
    A, C, D = 1.0, 0.5, 0.1
    blue, orange, gray = "#2B78D4", "#EB6834", "#555555"
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.plot([0, 1], [C + D, A + C + D], color=blue, lw=2.2,
            label="Reduzir dados ou épocas (modelo completo)", solid_capstyle="round")
    pts = sorted((REL[p][0], A * REL[p][0] + C * REL[p][1] + D, p) for p in WIDTHS)
    ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=orange, lw=2.2,
            marker="o", ms=6, mec="white", mew=1.2,
            label="Reduzir largura (todos os dados, 1 época)")
    ax.axhline(0.4, color=gray, ls="--", lw=1.2)
    labels = {1.0: "p = 1", 0.5: "p = 1/2", 0.25: "p = 1/4", 0.125: "p = 1/8"}
    offs = {1.0: (-0.04, 0.06), 0.5: (0.03, -0.06), 0.25: (0.03, -0.07),
            0.125: (0.03, -0.08)}
    for x, y, p in pts:
        dx, dy = offs[p]
        ax.text(x + dx, y + dy, labels[p], fontsize=8, color=gray,
                ha="right" if p == 1.0 else "left", va="center")
    ax.text(0.01, C + D - 0.08, "piso: C + D = 0,6", fontsize=8, color=gray)
    ax.text(0.97, 0.43, "orçamento da rodada b = 0,4", fontsize=8, color=gray,
            ha="right", va="bottom")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(0, 1.75)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(axis="y", color="#E3E3E3", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(colors=gray, labelsize=8, length=0)
    ax.set_xlabel("Computação relativa da rodada (η·L  ou  f(p))", color=gray, fontsize=9)
    ax.set_ylabel("Energia da rodada (unid. normalizadas)", color=gray, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- conteúdo
def build() -> list:
    s = []
    s += [
        Spacer(1, 30),
        P("Custo de Energia por Rodada em Aprendizado Federado", "title"),
        P("Como modelar a bateria para comparar redução de largura, de épocas e de dados",
          "sub"),
        P("Complemento ao documento “Heterogeneidade de Modelos e Consumo de Bateria” — "
          "18 de outubro de 2026, revisado em 19 de outubro de 2026", "sub"),
        Spacer(1, 12),
        note("<b>Objetivo.</b> O LeanFed desconta da bateria um custo que depende só do "
             "volume de dados. Isso não serve para comparar formas diferentes de economizar "
             "energia: reduzir a <b>largura</b> do modelo, o número de <b>épocas</b> ou a "
             "fração de <b>dados</b>. Este documento propõe um modelo de energia por rodada "
             "que trata os três ajustes da mesma forma, mostra o que cada um economiza e "
             "como calibrar os números. A seção 7 (nova nesta revisão) reescreve o modelo "
             "em <b>potência × tempo</b>, como no FedLoad, e mostra como obter a potência "
             "de cada dispositivo."),
        P("<b>Conteúdo</b>"),
        bullets([
            "1. Modelo de energia de uma rodada",
            "2. O que cada ajuste reduz",
            "3. Exemplo numérico",
            "4. Números para a CNN do projeto",
            "5. Atualizar a bateria e escolher o ajuste",
            "6. Calibração",
            "7. Potência × tempo: o modelo do FedLoad <i>(novo)</i>",
            "8. Esboço de código",
            "9. Origem e limitações do modelo",
            "10. Referências",
        ]),
    ]

    # 1 ----------------------------------------------------------------
    s += [
        H1("1. Modelo de energia de uma rodada"),
        P("A energia que o cliente <i>e</i> gasta em uma rodada é separada em três partes. "
          "Com largura do submodelo <i>p</i>, <i>L</i> épocas locais e fração de dados η:"),
        eq(r"$E_e(p, L, \eta) = A_e \cdot f(p) \cdot \eta \cdot L \;+\; "
           r"C_e \cdot s(p) \;+\; D_e$", 15),
        P("O primeiro termo é a <b>computação</b>, o segundo a <b>comunicação</b> e o "
          "terceiro o <b>custo fixo</b> de participar da rodada. A seção 9 explica a origem "
          "de cada parte e as limitações do modelo."),
        table(["Termo", "Significado"], [
            ["A<sub>e</sub>", "Energia de <b>uma época</b> sobre todos os dados locais com "
             "o modelo <b>completo</b>."],
            ["f(p) = F(p)/F(1)", "Custo de computação relativo da largura <i>p</i> (FLOPs "
             "ou MACs por amostra)."],
            ["C<sub>e</sub>", "Energia para baixar e enviar o modelo <b>completo</b>. "
             "Depende da rede do dispositivo (Wi-Fi, 4G); enviar costuma gastar mais que "
             "baixar."],
            ["s(p) = S(p)/S(1)", "Tamanho relativo do submodelo (número de parâmetros)."],
            ["D<sub>e</sub>", "Custo fixo de participar: acordar o dispositivo, carregar o "
             "modelo, a “cauda” do rádio depois de transmitir."],
        ], [1, 3.7]),
        P("Os dois coeficientes principais podem ser escritos a partir de grandezas "
          "físicas:"),
        eq(r"$A_e = k_e \cdot 3F(1) \cdot n_e \qquad\qquad "
           r"C_e = \left(\varepsilon_e^{\downarrow} + \varepsilon_e^{\uparrow}\right) "
           r"\cdot 4 \cdot S(1)$", 15),
        bullets([
            "k<sub>e</sub>: energia por operação (MAC) no hardware do dispositivo.",
            "O fator 3 aproxima o custo do treino: forward + backward ≈ 3 × o forward.",
            "n<sub>e</sub>: número de amostras locais de treino.",
            "ε<sup>↓</sup><sub>e</sub>, ε<sup>↑</sup><sub>e</sub>: energia por byte recebido "
            "e enviado; o fator 4 é o número de bytes por parâmetro em float32.",
        ]),
        P("Os mesmos coeficientes também podem ser escritos como potência × tempo, que é a "
          "forma usada pelo FedLoad [7] e a mais fácil de medir; a seção 7 mostra a "
          "equivalência."),
        note("<b>Por que normalizar:</b> os coeficientes estão sempre definidos para o "
             "modelo completo, e os fatores relativos f(p) e s(p) fazem o ajuste para cada "
             "largura. Assim, o mesmo A<sub>e</sub> e C<sub>e</sub> servem para todas as "
             "larguras, e só as tabelas de f e s dependem da arquitetura."),
    ]

    # 2 ----------------------------------------------------------------
    s += [
        H1("2. O que cada ajuste reduz"),
        table(["Ajuste", "Computação", "Comunicação", "Custo fixo"], [
            ["Dados η", "∝ η", "não muda", "não muda"],
            ["Épocas L", "∝ L", "não muda", "não muda"],
            ["Largura p", "∝ f(p) ≈ p<super>2</super>", "∝ s(p) ≈ p<super>2</super>",
             "não muda"],
            ["Pular a rodada", "0", "0", "0"],
        ], [1, 1.3, 1.3, 1.3]),
        P("Daí saem três consequências, que podem virar hipóteses da dissertação:"),
        P("Consequência 1 — Em energia, dados e épocas são equivalentes", "h3"),
        P("Só o produto η·L (o total de amostras processadas) importa. A diferença entre "
          "eles é <b>estatística</b>, não energética: várias épocas sobre poucos dados não "
          "aprendem o mesmo que uma época sobre todos os dados. Por isso a comparação entre "
          "os ajustes deve ser feita <b>com a mesma energia gasta</b>, medindo a precisão "
          "(comparação “iso-energia”)."),
        P("Consequência 2 — Largura é o único ajuste que reduz a comunicação", "h3"),
        P("Reduzindo dados ou épocas, o cliente ainda baixa e envia o modelo inteiro, então "
          "nunca gasta menos que C<sub>e</sub> + D<sub>e</sub> por rodada. Com bateria muito "
          "baixa, ele precisa pular rodadas. Com a largura, o mínimo cai para "
          "C<sub>e</sub>·s(p<sub>min</sub>) + D<sub>e</sub>, e o cliente consegue continuar "
          "participando com pouca energia."),
        P("Consequência 3 — Quem vence depende de C<sub>e</sub>/A<sub>e</sub>", "h3"),
        P("C<sub>e</sub>/A<sub>e</sub> mede o peso da comunicação em relação à computação. "
          "Se a comunicação pesa mais (modelo grande, rede ruim), a largura tende a vencer. "
          "Se a computação pesa mais, os três ajustes economizam parecido, e a qualidade do "
          "aprendizado decide."),
    ]

    # 3 ----------------------------------------------------------------
    fig_path = TMP / "fig1.png"
    figure1(fig_path)
    fig_w = WIDTH * 0.95
    rel = REL

    def rnd(p):
        return 1.0 * rel[p][0], 0.5 * rel[p][1]

    c2, m2 = rnd(0.5)
    c4, m4 = rnd(0.25)
    s += [
        H1("3. Exemplo numérico"),
        P("Valores <b>hipotéticos</b>, em unidades normalizadas, só para ilustrar: "
          "A<sub>e</sub> = 1, C<sub>e</sub> = 0,5, D<sub>e</sub> = 0,1, <i>L</i> = 1, com os "
          "fatores f(p) e s(p) da CNN do projeto (seção 4). Uma rodada completa (p = 1, "
          "η = 1) custa 1 + 0,5 + 0,1 = 1,6."),
        KeepTogether([
            Image(str(fig_path), width=fig_w, height=fig_w * 3.5 / 6.3),
            Spacer(1, 4),
            P("Figura 1 — Energia de uma rodada em função da computação feita. A reta azul "
              "reduz dados ou épocas com o modelo completo; a curva laranja reduz a largura "
              "usando todos os dados. A reta tracejada é um orçamento de 0,4 por rodada. "
              "Reduzindo só dados ou épocas, a energia nunca fica abaixo de C + D = 0,6; "
              "reduzindo a largura, ela se aproxima de D = 0,1.", "caption"),
        ]),
        table(["Orçamento b", "Só dados (p = 1)", "Só largura (η = 1)"], [
            ["0,8", "η* = (0,8 − 0,5 − 0,1) / 1 = <b>0,2</b>: modelo inteiro com 20% dos "
             "dados",
             f"p = 1/2 custa {br(c2, 3)} + {br(m2, 3)} + 0,1 = {br(c2 + m2 + 0.1, 3)} ≤ 0,8 "
             "→ <b>p = 1/2</b> com todos os dados"],
            ["0,4", "η* = (0,4 − 0,6) / 1 &lt; 0 → <b>não cabe</b>: o cliente pula a rodada",
             f"p = 1/4 custa {br(c4, 3)} + {br(m4, 3)} + 0,1 = {br(c4 + m4 + 0.1, 3)} ≤ 0,4 "
             "→ <b>p = 1/4</b>, continua participando"],
        ], [0.8, 2, 2]),
        P("Com orçamento 0,8, a pergunta de pesquisa aparece de forma concreta: com a mesma "
          "energia, o que contribui mais para o modelo global — 20% dos dados com o modelo "
          "inteiro, ou todos os dados com metade da largura? Com orçamento 0,4, só a largura "
          "permite participar."),
    ]

    # 4 ----------------------------------------------------------------
    layer_rows = []
    c1w, c2w, c3w, hw = 32, 64, 128, 256
    layers = [
        ("<font name='DejaVuMono'>conv1</font> (3→32)", 32 * 32 * c1w * 3 * 9,
         3 * c1w * 9 + c1w, "p"),
        ("<font name='DejaVuMono'>conv2</font> (32→64)", 16 * 16 * c2w * c1w * 9,
         c1w * c2w * 9 + c2w, "p<super>2</super>"),
        ("<font name='DejaVuMono'>conv3</font> (64→128)", 8 * 8 * c3w * c2w * 9,
         c2w * c3w * 9 + c3w, "p<super>2</super>"),
        ("<font name='DejaVuMono'>fc1</font> (2048→256)", c3w * 16 * hw,
         c3w * 16 * hw + hw, "p<super>2</super>"),
        ("<font name='DejaVuMono'>fc2</font> (256→10)", hw * 10, hw * 10 + 10, "p"),
    ]
    for name, m, n, law in layers:
        layer_rows.append([name, br(m, 0), br(100 * m / MACS1, 1) + "%", br(n, 0),
                           br(100 * n / PARAMS1, 1) + "%", law])
    layer_rows.append(["<b>Total</b>", f"<b>{br(MACS1, 0)}</b>", "100%",
                       f"<b>{br(PARAMS1, 0)}</b>", "100%", ""])
    frac = {1.0: "1", 0.5: "1/2", 0.25: "1/4", 0.125: "1/8"}
    s += [
        H1("4. Números para a CNN do projeto"),
        P("Contagem para o <font name='DejaVuMono'>Net</font> atual de "
          "<font name='DejaVuMono'>fedavg_app/task.py</font> (entrada 32×32), conferida "
          "executando o código da seção 8. No forward, são "
          f"<b>{br(MACS1 / 1e6, 2)} M MACs</b> por amostra; o modelo tem "
          f"<b>{round(PARAMS1 / 1e3)} mil parâmetros</b> "
          f"({br(4 * PARAMS1 / 1e6, 1)} MB em float32)."),
        table(["Camada", "MACs", "% MACs", "Parâmetros", "% parâm.", "Escala com"],
              layer_rows, [1.35, 1, 0.65, 1, 0.65, 0.9]),
        table(["Largura p", "f(p) — computação", "s(p) — comunicação"],
              [[frac[p], br(100 * REL[p][0], 1) + "%", br(100 * REL[p][1], 1) + "%"]
               for p in WIDTHS], [0.8, 1.8, 1.8]),
        P("O valor não é exatamente p<super>2</super> porque a "
          "<font name='DejaVuMono'>conv1</font> (3 canais de entrada fixos) e a "
          "<font name='DejaVuMono'>fc2</font> (10 saídas fixas) escalam só com <i>p</i>."),
        note("<b>Assimetria importante:</b> a <font name='DejaVuMono'>fc1</font> concentra "
             "<b>85% dos parâmetros</b>, mas só <b>5% da computação</b>; as convoluções "
             "concentram quase toda a computação, mas poucos parâmetros. Ou seja, nesse "
             "modelo a comunicação e a computação vêm de camadas diferentes, e a proporção "
             "C<sub>e</sub>/A<sub>e</sub> muda bastante de arquitetura para arquitetura. Vale "
             "testar pelo menos dois modelos (por exemplo, esta CNN e uma ResNet-18, que é "
             "dominada por convoluções)."),
    ]

    # 5 ----------------------------------------------------------------
    s += [
        H1("5. Atualizar a bateria e escolher o ajuste"),
        H2("5.1 Atualização"),
        P("A cada rodada em que o cliente participa:"),
        eq(r"$B_e \leftarrow B_e - E_e(p, L, \eta)$"),
        P("Para mais realismo, desconte também um consumo pequeno em repouso "
          "(δ<sub>e</sub>) nas rodadas em que ele não participa e, ao usar traces reais, "
          "some a recarga quando o dispositivo estiver carregando."),
        H2("5.2 Escolha do ajuste"),
        P("O servidor parte do orçamento da rodada e inverte a fórmula:"),
        eq(r"$b = \frac{B_e}{\mathrm{rodadas\ restantes}}$"),
        bullets(["<b>Só dados</b> (LeanFed corrigido — o original ignora C<sub>e</sub> e "
                 "D<sub>e</sub>):"]),
        eq(r"$\eta^* = \frac{b - C_e - D_e}{A_e \cdot L}$"),
        bullets(["<b>Só épocas:</b>"]),
        eq(r"$L^* = \left\lfloor \frac{b - C_e - D_e}{A_e \cdot \eta} \right\rfloor$"),
        P("Épocas são números inteiros, então a granularidade é ruim. Considere usar "
          "<b>passos locais</b> (número de batches) no lugar de épocas."),
        bullets([
            "<b>Só largura:</b> a maior <i>p</i> tal que E<sub>e</sub>(p, L, η) ≤ b.",
            "<b>Combinado:</b> testar as combinações de (p, η) que cabem no orçamento e "
            "escolher por uma regra — por exemplo, reduzir a largura até p<sub>min</sub> e "
            "só depois cortar dados.",
        ]),
        P("Nos três primeiros casos, se nada cabe no orçamento, o cliente pula a rodada. "
          "Quando pula, a energia não gasta aumenta o orçamento das rodadas seguintes."),
    ]

    # 6 ----------------------------------------------------------------
    s += [
        H1("6. Calibração"),
        H2("6.1 O problema da não-linearidade"),
        P("O modelo k<sub>e</sub>·f(p) supõe que a energia por operação é constante. Na "
          "prática, modelos pequenos aproveitam mal a GPU (overhead de lançamento de "
          "kernels, unidades ociosas). Por isso <i>p</i> = 1/4 economiza <b>menos</b> que os "
          f"{br(100 * REL[0.25][0], 1)}% da tabela sugerem. A solução é medir a energia por "
          "amostra em cada largura e usar a tabela medida f<sub>med</sub>(p) no lugar de "
          "f(p)."),
        H2("6.2 Opção (a): fórmula analítica com constantes medidas — recomendada"),
        P("Faça uma etapa de perfilamento separada, fora dos experimentos:"),
        bullets([
            "Meça a potência ociosa da GPU por alguns segundos.",
            "Para cada largura <i>p</i>, rode <b>um único</b> processo de treino por um "
            "número fixo de batches.",
            "Leia o contador de energia do NVML antes e depois "
            "(<font name='DejaVuMono'>nvmlDeviceGetTotalEnergyConsumption</font>, em mJ, "
            "disponível em GPUs a partir da geração Volta).",
            "Desconte potência ociosa × tempo e divida pelo número de amostras: energia por "
            "amostra e(p).",
            "Repita algumas vezes e use a média; f<sub>med</sub>(p) = e(p)/e(1) entra na "
            "fórmula no lugar de f(p).",
        ], numbered=True),
        P("O resultado é determinístico durante os experimentos e, portanto, reproduzível. "
          "Os parâmetros de comunicação (energia por byte, banda) não podem ser medidos na "
          "GPU: tire-os de medições publicadas para smartphones ou de medição própria, e "
          "passe-os por cliente no <font name='DejaVuMono'>--node-config</font>. A seção 7.6 "
          "mostra como usar o mesmo perfilamento para obter a potência em watts."),
        H2("6.3 Opção (b): potência × tempo emulado"),
        eq(r"$E_{comp} = P_e \cdot t_{sim} \qquad\qquad "
           r"E_{comm} = P_e^{\mathrm{rádio}} \cdot \frac{\mathrm{bytes}}{\mathrm{banda}_e}$"),
        P("É a forma usada pelo FedLoad [7] (seção 7). Reaproveita o "
          "<font name='DejaVuMono'>sim_time</font> que o "
          "<font name='DejaVuMono'>train()</font> já calcula e captura a não-linearidade "
          "sozinha. O problema é que, no setup atual, <b>todos os clientes dividem a mesma "
          "GPU</b>: o tempo medido inclui a disputa com os outros containers, varia entre "
          "execuções e atrapalha a reprodutibilidade. A seção 7.6 resolve isso calculando o "
          "tempo a partir do perfilamento, em vez de medi-lo durante o experimento."),
        note("<b>Atenção:</b> o NVML não separa a energia de cada container. Medições feitas "
             "durante um experimento com vários clientes misturam o consumo de todos; por "
             "isso a calibração deve ser feita em perfilamento separado, uma configuração "
             "por vez."),
    ]

    # 7 (novo) ---------------------------------------------------------
    t75, t25, t5 = (BITS1 / (b * 1e6) for b in (75, 25, 5))
    pr = 1.8  # W, potência de rádio usada no exemplo
    s += [
        H1("7. Potência × tempo: o modelo do FedLoad"),
        P("O FedLoad [7] (IEEE WCNC 2026) escolhe, a cada rodada, a taxa de poda de cada "
          "cliente com um bandit contextual (LinUCB). Para isso, modela a energia de uma "
          "rodada com a mesma estrutura deste documento, mas escrita em grandezas físicas: "
          "<b>potência (W) × tempo (s)</b>. Esta seção mostra a equivalência, o que o artigo "
          "define (e não define) sobre a potência, e como obter os números."),
        H2("7.1 O modelo do artigo"),
        P("Tempos de computação, upload e download (equação 2 e texto anterior a ela) e "
          "energia da rodada (equação 5). A duração da rodada síncrona é o tempo do cliente "
          "mais lento (equação 4):"),
        eq(r"$t^{\mathrm{cmp}}_e = \frac{\tau\, c_e\, |D_e|}{f_e} \qquad "
           r"t^{\mathrm{up}}_e = \frac{\Gamma}{B^{\mathrm{up}}_e} \qquad "
           r"t^{\mathrm{down}}_e = \frac{\Gamma}{B^{\mathrm{down}}_e}$"),
        eq(r"$E_e = P^{\mathrm{cmp}}_e\, t^{\mathrm{cmp}}_e + P^{\mathrm{up}}_e\, "
           r"t^{\mathrm{up}}_e + P^{\mathrm{down}}_e\, t^{\mathrm{down}}_e \qquad\qquad "
           r"T = \max_{e \in S}\; t_e$"),
        table(["Símbolo", "Significado no artigo"], [
            ["τ", "Número de iterações locais (no artigo, multiplica o dataset inteiro, ou "
             "seja, funciona como épocas)."],
            ["c<sub>e</sub>, f<sub>e</sub>", "Ciclos de CPU por amostra e frequência da CPU."],
            ["|D<sub>e</sub>|", "Número de amostras locais."],
            ["Γ, B<super>up</super><sub>e</sub>, B<super>down</super><sub>e</sub>",
             "Tamanho do submodelo e vazão média de upload e download."],
            ["P<super>cmp</super><sub>e</sub>, P<super>up</super><sub>e</sub>, "
             "P<super>down</super><sub>e</sub>",
             "Potência durante treino, upload e download. No artigo: e<sub>i,cmp</sub>, "
             "e<sub>i,up</sub>, e<sub>i,down</sub>. Aqui se usa P, como na seção 6.3, para "
             "não confundir com o índice do cliente <i>e</i>."],
        ], [1.3, 3.4]),
        H2("7.2 Equivalência com este documento"),
        eq(r"$A_e = P^{\mathrm{cmp}}_e \cdot \frac{c_e\, n_e}{f_e} \qquad "
           r"C_e = \Gamma(1) \left( \frac{P^{\mathrm{down}}_e}{B^{\mathrm{down}}_e} + "
           r"\frac{P^{\mathrm{up}}_e}{B^{\mathrm{up}}_e} \right) \qquad D_e = 0$"),
        table(["Este documento", "FedLoad [7]"], [
            ["A<sub>e</sub> (uma época, modelo completo)",
             "P<super>cmp</super><sub>e</sub> · c<sub>e</sub> n<sub>e</sub> / f<sub>e</sub> "
             "(potência × tempo de uma época)"],
            ["ε<sup>↓</sup><sub>e</sub>, ε<sup>↑</sup><sub>e</sub> (energia por byte)",
             "P<super>down</super><sub>e</sub> / B<super>down</super><sub>e</sub>, "
             "P<super>up</super><sub>e</sub> / B<super>up</super><sub>e</sub> "
             "(potência ÷ vazão)"],
            ["C<sub>e</sub> · s(p)", "Γ(p) · (P<super>down</super>/B<super>down</super> + "
             "P<super>up</super>/B<super>up</super>)"],
            ["D<sub>e</sub> (custo fixo)", "Não existe (vale 0)."],
            ["L · η (amostras processadas)", "τ · |D<sub>e</sub>|"],
        ], [2, 2.7]),
        P("O FedLoad não acrescenta termos: é um caso particular deste modelo, sem o custo "
          "fixo. A vantagem da forma potência × tempo é prática. Os parâmetros têm unidade "
          "física e podem ser medidos ou tirados de fichas técnicas. Além disso, o mesmo "
          "conjunto de parâmetros dá também o <b>tempo</b> da rodada, de modo que tempo e "
          "energia ficam coerentes e podem ser usados juntos (por exemplo, na "
          "<font name='DejaVuMono'>straggler-policy</font> que o projeto já tem)."),
        P("<b>Exemplo — comunicação da CNN do projeto.</b> O modelo completo tem "
          f"{br(PARAMS1, 0)} parâmetros, ou {br(BITS1 / 1e6, 1)} Mb em float32. Com potência "
          f"de rádio de {br(pr, 1)} W nos dois sentidos e a mesma vazão de upload e "
          "download, nas três faixas de rede da Tabela I do FedLoad:"),
        table(["Vazão", "Tempo para baixar (ou enviar)", "C<sub>e</sub> (baixar + enviar)",
               "C<sub>e</sub> · s(1/4)"], [
            [f"{b} Mbps", f"{br(t, 2)} s", f"{br(2 * pr * t, 2)} J",
             f"{br(2 * pr * t * REL[0.25][1], 2)} J"]
            for b, t in ((75, t75), (25, t25), (5, t5))
        ], [1, 1.6, 1.4, 1.2]),
        P("Para comparar com A<sub>e</sub> e saber qual termo domina "
          "(C<sub>e</sub>/A<sub>e</sub>, consequência 3), falta o tempo de treino por "
          "amostra, que sai do perfilamento da seção 6.2."),
        H2("7.3 O que é e<sub>i,cmp</sub> — e o que o artigo não diz"),
        P("<b>Definição.</b> A única definição está logo depois da equação 5: "
          "<i>“the power consumption rate (in Joules per second) during local training "
          "phase”</i>. Joules por segundo é watt: e<sub>i,cmp</sub> é a <b>potência média do "
          "dispositivo enquanto treina</b>. e<sub>i,up</sub> e e<sub>i,down</sub> são a "
          "potência durante a transmissão e a recepção."),
        P("<b>Origem dos valores: não explicada.</b> Na simulação, a potência é só uma "
          "constante sorteada por cliente (Tabela I do artigo), sem fonte ou justificativa. "
          "Na equação 9, os clientes enviam esses valores ao servidor como contexto do "
          "bandit, mas o artigo não diz se são medidos, estimados ou tirados da ficha "
          "técnica. Os valores ficam na faixa de um SoC de celular ou de uma placa embarcada "
          "sob carga; isso é uma leitura, não algo escrito no texto."),
        table(["Parâmetro", "Tabela I do FedLoad", "Observação"], [
            ["e<sub>i,cmp</sub>", "U({7, 4, 3}) W", "Três níveis de potência de treino."],
            ["e<sub>i,up</sub>, e<sub>i,down</sub>", "U({2; 1,8; 1,7}) W",
             "Sorteados separadamente."],
            ["B<super>up</super>, B<super>down</super>", "U(1, U({75, 25, 5})) Mbps",
             "Cada cliente tem uma faixa (teto); a vazão de cada rodada é sorteada entre "
             "1 Mbps e o teto."],
            ["f<sub>i</sub>", "U(1, 2) “Hz”", "Provavelmente GHz."],
            ["c<sub>r,i</sub>", "U(5, 30) ciclos por amostra",
             "Não depende da taxa de poda (ver 7.4)."],
            ["τ", "5", "Iterações locais."],
        ], [1.2, 1.6, 2.2]),
        note("<b>As unidades da Tabela I não fecham.</b> Com τ = 5, c ≈ 17,5, |D| ≈ 1.667 "
             "(CIFAR-10 dividido entre 30 clientes) e f ≈ 1,5, a equação 2 dá "
             "t<super>cmp</super> ≈ 97.000 s por rodada, cerca de 27 horas. Os números servem "
             "como <b>faixas relativas de heterogeneidade</b>, não como valores físicos."),
        H2("7.4 Cuidados ao reaproveitar o modelo"),
        bullets([
            "<b>A largura não aparece nas fórmulas.</b> O artigo diz que a taxa de poda "
            "determina o tamanho e o custo do submodelo, mas na Tabela I c<sub>r,i</sub> é "
            "sorteado sem depender dela. Ao implementar, use c(p) = c · f(p) e "
            "Γ(p) = Γ(1) · s(p). O texto chama a taxa de “proporção de parâmetros mantidos”, "
            "mas a poda mantém ⌊n<sub>l</sub> · p⌋ neurônios por camada, o que dá "
            "parâmetros ≈ p<super>2</super> (seção 4).",
            "<b>Unidades da Tabela I</b> (quadro acima): calibre os números com medições "
            "próprias.",
            "<b>Sem custo fixo</b> D<sub>e</sub> nem consumo em repouso.",
            "<b>Potência independente da frequência.</b> P<super>cmp</super> e f são "
            "sorteados separadamente. Fisicamente, a potência cresce com a frequência "
            "(7.5, item 3).",
            "<b>Não há bateria.</b> A energia é um termo da recompensa do bandit (peso "
            "w<sub>2</sub> = 0,1), não uma restrição: não há capacidade, estado de carga nem "
            "saída de clientes. Para este documento, isso significa somar a atualização da "
            "seção 5.1. É também o que diferencia a proposta da dissertação do FedLoad, que "
            "serve como baseline.",
            "<b>Tudo simulado</b>, sem medição em dispositivo real, como no LeanFed.",
        ]),
        H2("7.5 Como obter a potência P<super>cmp</super>"),
        P("<b>1. Medir</b> — a melhor opção para a dissertação.", "h3free"),
        table(["Hardware", "Como medir"], [
            ["GPU NVIDIA (setup atual)", "<font name='DejaVuMono'>pynvml</font>: "
             "<font name='DejaVuMono'>nvmlDeviceGetPowerUsage</font> (mW, instantâneo) ou "
             "<font name='DejaVuMono'>nvmlDeviceGetTotalEnergyConsumption</font> (mJ "
             "acumulados, GPUs Volta ou mais novas)."],
            ["CPU Intel/AMD", "RAPL: <font name='DejaVuMono'>/sys/class/powercap/"
             "intel-rapl*/energy_uj</font>, ou as bibliotecas pyRAPL e CodeCarbon."],
            ["Raspberry Pi / Jetson", "Medidor USB de tensão × corrente; na Jetson, "
             "<font name='DejaVuMono'>tegrastats</font> ou os sensores INA internos."],
            ["Android", "<font name='DejaVuMono'>BatteryManager</font>: corrente × tensão."],
        ], [1.4, 3.3], split=True),
        P("A potência média é a energia medida dividida pelo tempo de treino. Para isolar o "
          "custo do treino, desconte a potência ociosa, como na seção 6.2. No setup atual, "
          "todos os clientes dividem a mesma GPU e o NVML mede a placa inteira: meça com "
          "<b>um cliente sozinho</b>."),
        P("<b>2. Ficha técnica</b>", "h3"),
        P("TDP ou potência típica sob carga do dispositivo que se quer emular. É simples, "
          "mas superestima, porque o treino raramente usa o TDP inteiro."),
        P("<b>3. Modelo analítico de CMOS</b>", "h3"),
        P("Vários trabalhos de FL em redes sem fio usam o modelo de potência dinâmica de "
          "CMOS, em que a energia por ciclo cresce com f<super>2</super> e o chip executa "
          "f ciclos por segundo:"),
        eq(r"$P^{\mathrm{cmp}}_e = \kappa\, f_e^3 \quad\Rightarrow\quad "
           r"E^{\mathrm{cmp}}_e = P^{\mathrm{cmp}}_e \cdot \frac{\tau\, c_e\, |D_e|}{f_e} "
           r"= \kappa\, \tau\, c_e\, |D_e|\, f_e^2$"),
        P("É exatamente a forma de Yang et al. [1] (seção 9.1). κ é a capacitância efetiva "
          "do chip; um valor comum na literatura é 10<super>−28</super>, mas confira a "
          "convenção no artigo de onde ele for tirado. Esse modelo amarra potência e "
          "velocidade: um cliente com o dobro da frequência termina em metade do tempo, mas "
          "gasta 4× mais energia por amostra. O FedLoad não faz essa ligação."),
        H2("7.6 Recomendação"),
        P("Combinar as opções 1 e 3, mantendo a reprodutibilidade da seção 6.2:"),
        bullets([
            "<b>Referência medida.</b> No perfilamento da seção 6.2, com um cliente sozinho, "
            "registre para cada largura a potência média P<sub>ref</sub> (W) e o tempo de "
            "treino por amostra t<sub>amostra</sub>(p) (s).",
            "<b>Heterogeneidade em torno da referência.</b> Para cada cliente, "
            "P<super>cmp</super><sub>e</sub> = P<sub>ref</sub> × um fator. O fator pode vir "
            "de faixas, como os níveis 7/4/3 W do FedLoad em proporção, ou do modelo de "
            "CMOS, P ∝ v<sub>e</sub><super>3</super>, se a potência deve acompanhar a "
            "velocidade v<sub>e</sub> (o <font name='DejaVuMono'>compute-speed</font>). "
            "Escolha uma das duas e declare.",
            "<b>Tempo calculado, não medido.</b> Use o tempo do perfilamento, dividido pela "
            "velocidade do cliente, em vez do <font name='DejaVuMono'>sim_time</font> medido "
            "durante o experimento, que inclui a disputa pela GPU (seção 6.3):",
        ], numbered=True),
        eq(r"$t^{\mathrm{cmp}}_e(p) = \frac{\eta\, L\, n_e\, t_{\mathrm{amostra}}(p)}{v_e} "
           r"\qquad\qquad E^{\mathrm{cmp}}_e = P^{\mathrm{cmp}}_e \cdot t^{\mathrm{cmp}}_e(p)$"),
        ListFlowable(
            [ListItem(P(t), leftIndent=16) for t in [
                "<b>Comunicação por faixas de rede.</b> Γ(p) = 4 · S(1) · s(p) bytes; cada "
                "cliente recebe uma faixa (75, 25 ou 5 Mbps) e a vazão de cada rodada é "
                "sorteada como U(1, faixa), com seed; potência de rádio entre 1,7 e 2 W, "
                "como no FedLoad.",
                "<b>Tudo pelo <font name='DejaVuMono'>--node-config</font>.</b> Potências, "
                "faixa de rede e bateria inicial geradas pelo "
                "<font name='DejaVuMono'>scripts/gen_compose.py</font>, como já acontece "
                "com o <font name='DejaVuMono'>compute-speed</font>.",
            ]],
            bulletType="1", bulletFormat="%s.", start=4, leftIndent=16,
            bulletFontName="DejaVu", bulletFontSize=9, spaceAfter=4,
        ),
        P("O resultado alimenta as fórmulas das seções 1 e 5 sem mudar nada nelas: "
          "A<sub>e</sub> = P<super>cmp</super><sub>e</sub> · n<sub>e</sub> · "
          "t<sub>amostra</sub>(1) / v<sub>e</sub>, f<sub>med</sub>(p) = "
          "t<sub>amostra</sub>(p) / t<sub>amostra</sub>(1), e C<sub>e</sub> como em 7.2. O "
          "código está na seção 8.5."),
        note("<b>Suposição a declarar:</b> escrever f<sub>med</sub> como razão de tempos "
             "supõe que a potência é a mesma em todas as larguras. Modelos pequenos podem "
             "puxar menos potência da GPU; o perfilamento da seção 6.2 mede energia e tempo "
             "ao mesmo tempo e permite conferir. Como bônus, o mesmo modelo dá o tempo da "
             "rodada, T = max t<sub>e</sub>, e permite reportar tempo e energia juntos, como "
             "o FedLoad faz na Tabela II."),
    ]

    # 8 ----------------------------------------------------------------
    s += [
        H1("8. Esboço de código"),
        H2("8.1 Contagem de MACs e parâmetros para qualquer largura"),
        P("Usa hooks do PyTorch, então funciona para qualquer arquitetura feita de "
          "<font name='DejaVuMono'>Conv2d</font> e <font name='DejaVuMono'>Linear</font>. Os "
          "números da seção 4 saíram desta função."),
        code('''
def macs_and_params(width: float) -> tuple[int, int]:
    net, macs = Net(width), 0
    def hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            macs += out.numel() * (m.in_channels // m.groups) * kh * kw
        elif isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features
    hs = [m.register_forward_hook(hook) for m in net.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    net(torch.zeros(1, 3, 32, 32))
    for h in hs:
        h.remove()
    return macs, sum(t.numel() for t in net.parameters())
'''),
        code(head="8.2 Energia prevista de uma rodada (planejamento no servidor)", text='''
def round_energy(prof: dict, width: float, epochs: int, frac: float) -> float:
    f = REL_COMPUTE[width]   # f(p), ou f_med(p) medido na calibração
    s = REL_PARAMS[width]    # s(p)
    return prof["A"] * f * frac * epochs + prof["C"] * s + prof["D"]
'''),
        code(head="8.3 Escolha da largura dentro do orçamento", text='''
WIDTHS = (1.0, 0.5, 0.25, 0.125)

def choose_width(prof: dict, battery: float, rounds_left: int, epochs: int = 1) -> float:
    budget = battery / rounds_left
    for p in WIDTHS:                                 # da maior para a menor
        if round_energy(prof, p, epochs, 1.0) <= budget:
            return p
    return 0.0                                       # não participa nesta rodada
'''),
        H2("8.4 Consistência entre planejamento e execução"),
        P("No cliente, o desconto feito batch a batch deve usar os mesmos coeficientes:"),
        bullets([
            "por batch: k<sub>e</sub> · 3F(p) · tamanho do batch (na forma da seção 7: "
            "P<super>cmp</super><sub>e</sub> · t<sub>amostra</sub>(p) / v<sub>e</sub> · "
            "tamanho do batch);",
            "ao baixar e ao enviar o modelo: a parte correspondente de "
            "C<sub>e</sub>·s(p);",
            "uma vez por rodada: D<sub>e</sub>.",
        ]),
        P("Assim, o que o servidor planeja e o que o cliente gasta seguem o mesmo modelo, e "
          "a diferença entre os dois vem só do que você introduzir de propósito — por "
          "exemplo, ruído na estimativa da bateria para testar a robustez da política."),
        H2("8.5 Coeficientes a partir de potência e tempo (seção 7)"),
        P("Converte as grandezas físicas de cada cliente no dicionário "
          "<font name='DejaVuMono'>prof</font> usado acima. O tempo vem do perfilamento, "
          "não do experimento."),
        code('''
def profile_from_physics(n_samples: int, t_sample: float, speed: float,
                         p_cmp: float, p_down: float, p_up: float,
                         bw_down: float, bw_up: float, n_params: int,
                         d_fixed: float = 0.0) -> dict:
    """W, s e bit/s -> coeficientes A, C, D (J) e tempos T, Tc (s) do modelo completo."""
    t_epoch = n_samples * t_sample / speed           # uma época, largura 1
    bits = 32 * n_params                             # float32
    t_comm = bits / bw_down + bits / bw_up
    return {
        "A": p_cmp * t_epoch,
        "C": bits * (p_down / bw_down + p_up / bw_up),
        "D": d_fixed,
        "T": t_epoch,
        "Tc": t_comm,
    }

def round_time(prof: dict, width: float, epochs: int, frac: float) -> float:
    return (prof["T"] * REL_COMPUTE[width] * frac * epochs
            + prof["Tc"] * REL_PARAMS[width])
'''),
    ]

    # 9 ----------------------------------------------------------------
    s += [
        H1("9. Origem e limitações do modelo"),
        P("O modelo não foi tirado de um artigo específico. Ele junta uma estrutura padrão "
          "da literatura de FL com restrição de energia (computação + comunicação) com dois "
          "acréscimos: os fatores de largura f(p) e s(p) e o custo fixo D<sub>e</sub>. Esta "
          "seção mostra de onde vem cada parte, o que é suposição e como validar."),
        H2("9.1 A base: computação + comunicação"),
        P("A decomposição aparece, por exemplo, em Yang et al. [1], equações (2), (3) e (5) "
          "do artigo (conferidas no PDF):"),
        eq(r"$E_k^C = \kappa\, I_k\, C_k\, D_k\, f_k^2 \qquad\qquad "
           r"E_k^T = t_k\, p_k, \quad t_k\, r_k \geq s$", 15),
        bullets([
            "<b>Computação:</b> κ é uma constante do chip, I<sub>k</sub> o número de "
            "iterações locais, C<sub>k</sub> os ciclos de CPU por amostra, D<sub>k</sub> o "
            "número de amostras e f<sub>k</sub> a frequência da CPU.",
            "<b>Transmissão:</b> potência × tempo, onde o tempo é o tamanho do modelo "
            "<i>s</i> dividido pela taxa r<sub>k</sub>.",
        ]),
        P("O modelo deste documento é a mesma estrutura, reescrita:"),
        table(["Yang et al. [1]", "Este documento"], [
            ["I<sub>k</sub> (iterações locais)", "<i>L</i> (épocas)"],
            ["D<sub>k</sub> (amostras)", "η · n<sub>e</sub>"],
            ["C<sub>k</sub> (ciclos por amostra)", "∝ 3F(p) — é aqui que entra a largura"],
            ["κ f<sub>k</sub><super>2</super> (energia por ciclo)", "k<sub>e</sub>"],
            ["p<sub>k</sub> · s / r<sub>k</sub>", "C<sub>e</sub> · s(p) — energia por byte × "
             "tamanho do modelo"],
        ], [1, 1]),
        P("O FedLoad [7] usa a mesma estrutura, escrita como potência × tempo para a "
          "computação também; a equivalência está na seção 7.2."),
        note("<b>Detalhe relevante:</b> Yang et al. assumem que o tamanho do modelo é "
             "<b>igual para todos os usuários</b> (“the dimensions of the local FL model are "
             "fixed for all users”). Com submodelos, o <i>s</i> passa a depender da largura "
             "— é exatamente a lacuna explorada aqui. Eles também desprezam o download, "
             "supondo que a estação base transmite com muita potência; este documento inclui "
             "o download, mas ele pode ser desprezado com a mesma justificativa."),
        P("9.2 O que foi acrescentado", "h2free"),
        table(["Peça", "De onde veio", "Grau de confiança"], [
            ["<b>f(p) e s(p)</b>", "As mesmas métricas de custo do HeteroFL [5]: FLOPs como "
             "medida de computação e número de parâmetros como medida de comunicação, aqui "
             "divididas pelo valor do modelo completo (detalhes em 9.3)",
             "A contagem é exata. A suposição <b>energia ∝ FLOPs</b> é fraca (9.4, item 1)"],
            ["<b>Fator 3</b> (treino ≈ 3× forward)", "Aproximação padrão: o backward custa "
             "cerca de 2× o forward", "Boa aproximação; irrelevante após normalizar, pois se "
             "cancela em f(p)"],
            ["<b>Custo fixo D<sub>e</sub></b>", "Medições em celulares: o rádio permanece em "
             "alta potência após cada transferência (“tail energy”) [2]",
             "Fenômeno bem documentado em 3G/GSM; o <b>valor</b> em redes atuais é "
             "desconhecido"],
            ["<b>Normalização</b>", "Escolha de modelagem, para separar o que é do "
             "dispositivo (A<sub>e</sub>, C<sub>e</sub>) do que é da arquitetura (f, s)",
             "É só reescrita; não muda o modelo"],
            ["<b>Potência × tempo</b> (seção 7)", "FedLoad [7], equações 2 e 5",
             "Mesma estrutura; os valores da Tabela I do artigo não têm fonte e as unidades "
             "não fecham"],
            ["<b>Valores do exemplo</b> (A = 1, C = 0,5, D = 0,1)", "Inventados, só para "
             "ilustrar", "Nenhum — não usar como resultado"],
        ], [1.1, 1.8, 1.6]),
        H2("9.3 f(p) e s(p): as métricas de custo do HeteroFL"),
        P("<b>Como o HeteroFL cria os submodelos.</b> O HeteroFL [5] reduz a <b>largura</b> "
          "das camadas ocultas (o número de canais), mantendo a profundidade. Para uma "
          "camada oculta com pesos W<sub>g</sub> de dimensão d<sub>g</sub> × k<sub>g</sub> "
          "(canais de saída × canais de entrada), o submodelo do nível <i>l</i> mantém uma "
          "fração r<super>l−1</super> dos canais de saída <b>e</b> de entrada. Como as duas "
          "dimensões encolhem, o número de pesos cai com o quadrado (seção 3.1 do artigo):"),
        eq(r"$|W_l| = r^{2(l-1)}\, |W_g|$", 15),
        P("Os experimentos usam r = 0,5 e cinco níveis, de ‘a’ (modelo completo) a ‘e’, com "
          "larguras relativas 1; 0,5; 0,25; 0,125 e 0,0625."),
        note("<b>Cuidado com a notação:</b> no HeteroFL, a letra <i>p</i> é o <b>índice do "
             "nível</b> e <i>r</i> é a razão de encolhimento. Neste documento, <i>p</i> é a "
             "<b>própria largura relativa</b>: o nível <i>l</i> do HeteroFL corresponde a "
             "p = r<super>l−1</super>, e a lei acima vira s(p) ≈ p<super>2</super>."),
        P("<b>Como o HeteroFL mede o custo.</b> As tabelas de resultados trazem, para cada "
          "modelo ou combinação de modelos, três colunas de custo: <b>Parameters</b> (número "
          "de parâmetros), <b>FLOPs</b> (operações de um forward) e <b>Space (MB)</b> "
          "(memória ocupada pelos parâmetros). Há também uma coluna <b>Ratio</b>, com o "
          "número de parâmetros relativo ao maior modelo da combinação. Os autores dizem "
          "usar essas colunas para indicar os requisitos de computação e de comunicação dos "
          "métodos. Exemplo da Tabela 1 do artigo (MNIST), comparando o modelo completo com "
          "o de meia largura:"),
        table(["Modelo", "Largura", "Parameters", "FLOPs", "Space (MB)"], [
            ["‘a’", "1", "1,6 M", "80,5 M", "5,94"],
            ["‘b’", "0,5", "391 K", "20,5 M", "1,49"],
            ["<b>‘b’ / ‘a’</b>", "", "≈ 0,24", "≈ 0,25", "≈ 0,25"],
        ], [1, 0.8, 1.4, 1.4, 1.4]),
        P("Reduzir a largura à metade reduz parâmetros e FLOPs a cerca de 1/4, como prevê a "
          "lei quadrática."),
        P("<b>Como isso vira f(p) e s(p).</b> As duas funções são essas mesmas colunas, "
          "divididas pelo valor do modelo completo:"),
        eq(r"$f(p) = \frac{\mathrm{FLOPs}(p)}{\mathrm{FLOPs}(1)} \qquad\qquad "
           r"s(p) = \frac{\mathrm{Parâmetros}(p)}{\mathrm{Parâmetros}(1)}$"),
        bullets([
            "<b>f(p)</b> representa a computação: FLOPs por amostra no forward. O treino "
            "custa cerca de 3× isso, mas o fator se cancela na divisão.",
            "<b>s(p)</b> representa a comunicação: em cada rodada o cliente baixa e envia "
            "exatamente os parâmetros do seu submodelo, então os bytes transmitidos são "
            "proporcionais ao número de parâmetros (4 bytes cada em float32). A coluna Space "
            "(MB) do HeteroFL é essa mesma grandeza em megabytes.",
        ]),
        P("<b>Por que não dá exatamente p<super>2</super> no modelo inteiro.</b> A lei "
          "quadrática vale para camadas ocultas, em que entrada e saída encolhem juntas. A "
          "primeira camada tem entrada fixa (3 canais de cor) e a última tem saída fixa (o "
          "número de classes), então essas duas escalam só com <i>p</i>. Na CNN do projeto, "
          f"isso dá s(1/2) = {br(100 * REL[0.5][1], 1)}% e f(1/2) = "
          f"{br(100 * REL[0.5][0], 1)}% em vez de 25% (seção 4). Por isso o documento usa a "
          "<b>contagem exata</b> (código da seção 8.1) em vez da fórmula p<super>2</super>."),
        P("<b>O que o HeteroFL não faz.</b> O artigo reporta FLOPs e parâmetros como "
          "indicadores de custo, mas não mede tempo nem energia. O passo de “FLOPs e "
          "parâmetros” para “joules” é o que este modelo acrescenta, com os coeficientes "
          "A<sub>e</sub> e C<sub>e</sub>, e é exatamente esse passo que pode falhar (item 1 "
          "de 9.4) e que a calibração verifica."),
        H2("9.4 Onde o modelo pode falhar"),
        bullets([
            "<b>Energia proporcional a FLOPs é a maior fraqueza.</b> Yang, Chen e Sze [3] "
            "mostram que reduzir computação ou tamanho do modelo não reduz a energia na "
            "mesma proporção, porque boa parte dela vai para mover dados na memória. Logo, "
            "f(p) provavelmente <b>superestima</b> a economia de submodelos pequenos. Por "
            "isso a calibração (tabela medida f<sub>med</sub>(p), seção 6) é parte "
            "essencial do modelo, não um detalhe.",
            "<b>Frequência da CPU fixa.</b> Yang et al. [1] otimizam a frequência "
            "f<sub>k</sub>, e a energia por ciclo cresce com f<sub>k</sub><super>2</super>. "
            "Tratar k<sub>e</sub> como constante equivale a supor que o dispositivo sempre "
            "roda na mesma frequência. Se a heterogeneidade de velocidade vier de "
            "frequências diferentes, use P ∝ f<super>3</super> (seção 7.5, item 3).",
            "<b>Potência de base durante a rodada.</b> Enquanto treina, o dispositivo gasta "
            "uma potência de base (CPU acordada, memória) proporcional ao <b>tempo</b>, não "
            "às operações. Se o tempo cresce com o trabalho, essa parte fica embutida em "
            "A<sub>e</sub>; se não cresce, aparece como custo fixo extra. Medir de ponta a "
            "ponta resolve isso automaticamente.",
            "<b>Bateria linear.</b> O modelo supõe que a carga diminui exatamente pela "
            "energia consumida. Baterias reais têm efeitos não-lineares (perdem capacidade "
            "com corrente alta, se recuperam em repouso, variam com a temperatura). A "
            "maioria dos artigos de FL ignora isso; para um mestrado é uma simplificação "
            "aceitável, desde que declarada.",
        ], numbered=True),
        note("<b>A conclusão qualitativa resiste a essas limitações:</b> reduzir dados ou "
             "épocas não diminui a comunicação nem o custo fixo, então existe um piso de "
             "C + D. Isso vale para quaisquer C, D &gt; 0, sem depender de números. O que muda "
             "com os itens 1 a 4 é <b>o quanto</b> a largura vence."),
        H2("9.5 Como validar"),
        P("Isso pode virar uma seção da dissertação (“validação do modelo de energia”). Com "
          "o perfilamento por NVML da seção 6, teste três previsões:"),
        bullets([
            "<b>Energia × fração de dados η</b> (com p = 1): a fórmula prevê uma <b>reta</b>. "
            "A inclinação estima A<sub>e</sub> e o intercepto estima o custo fixo.",
            "<b>Energia × épocas L:</b> também uma reta, com a mesma inclinação por amostra "
            "processada.",
            "<b>Energia × largura p:</b> compare a energia medida com f(p). Se a razão "
            "f<sub>med</sub>(p)/f(p) estiver longe de 1 para <i>p</i> pequeno, a fraqueza do "
            "item 1 está confirmada, e a tabela medida substitui f(p).",
        ], numbered=True),
        P("Faça uma regressão linear e reporte o R². Se as retas se ajustarem bem, o modelo "
          "está <b>validado empiricamente no hardware usado</b>, o que é mais do que LeanFed "
          "[4], EAFL [6] e FedLoad [7] fazem. A comunicação não pode ser medida na GPU: cite "
          "medições da literatura, mantenha C<sub>e</sub> como parâmetro e faça uma análise "
          "de sensibilidade, variando C<sub>e</sub>/A<sub>e</sub> e mostrando como a "
          "conclusão muda."),
    ]

    # 10 ---------------------------------------------------------------
    def link(url):
        return f"<link href='{url}' color='#1F4E8C'><u>{url}</u></link>"

    refs = [
        "Z. Yang, M. Chen, W. Saad, C. S. Hong, M. Shikh-Bahaei. Energy Efficient Federated "
        "Learning Over Wireless Communication Networks. IEEE Transactions on Wireless "
        "Communications. " + link("https://arxiv.org/abs/1911.02417"),
        "N. Balasubramanian, A. Balasubramanian, A. Venkataramani. Energy Consumption in "
        "Mobile Phones: A Measurement Study and Implications for Network Applications. ACM "
        "IMC 2009. " + link("https://people.cs.umass.edu/~arun/papers/TailEnder.pdf"),
        "T.-J. Yang, Y.-H. Chen, V. Sze. Designing Energy-Efficient Convolutional Neural "
        "Networks Using Energy-Aware Pruning. CVPR 2017. "
        + link("https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_"
               "Energy-Efficient_Convolutional_CVPR_2017_paper.html"),
        "R. Pereira, C. J. Vaca-Rubio, L. Blanco. Learn More by Using Less: Distributed "
        "Learning with Energy-Constrained Devices (LeanFed). "
        + link("https://arxiv.org/abs/2412.02289"),
        "E. Diao, J. Ding, V. Tarokh. HeteroFL: Computation and Communication Efficient "
        "Federated Learning for Heterogeneous Clients. ICLR 2021. "
        + link("https://arxiv.org/abs/2010.01264"),
        "EAFL (Arouj &amp; Abdelmoniem, 2022) e demais trabalhos citados — ver o documento "
        "“Heterogeneidade de Modelos e Consumo de Bateria”.",
        "B. Martins, E. Samikwa, T. Braun, D. Rosário, E. Cerqueira, L. Villas. FedLoad: "
        "Adaptive Partial Training for Model Heterogeneous Federated Learning. IEEE WCNC "
        "2026. DOI " + link("https://doi.org/10.1109/WCNC65185.2026.11555362"),
    ]
    s += [H1("10. Referências")]
    s += [P(f"[{i}] {r}", "ref") for i, r in enumerate(refs, 1)]
    return s


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.8 * inch, bottomMargin=0.9 * inch,
        title="Custo de Energia por Rodada em FL",
        subject="Modelo de energia para largura, épocas e dados",
        author="AprendFed",
    )
    doc.build(build(), onFirstPage=footer, onLaterPages=footer)
    print(f"{OUT} gerado.")


if __name__ == "__main__":
    main()
