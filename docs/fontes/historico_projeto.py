#!/usr/bin/env python3
"""Gera docs/historico_projeto.pdf (histórico do projeto de 15 a 21/10/2026).

Uso:
    python3 docs/fontes/historico_projeto.py [saida.pdf]

Requer reportlab, matplotlib e as fontes DejaVu. Os números vêm das runs em
outputs/ (ids no Apêndice A) e de graficos/seeds*/runs_seeds.csv; estão escritos
aqui para o PDF não depender de outputs/, que não é versionado.
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "historico_projeto.pdf"
TMP = Path(tempfile.mkdtemp(prefix="historico_"))

# ---------------------------------------------------------------- estilo
# Igual ao de custo_energia_rodada.py, para os documentos da pasta combinarem.
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
    canvas.drawString(MARGIN, y, "Histórico do projeto AprendFed — 15 a 21 de outubro de 2026")
    canvas.drawRightString(A4[0] - MARGIN, y, str(doc.page))
    canvas.restoreState()


# ---------------------------------------------------------------- dados
# Acurácia média no teste por rodada (0 a 20), média das seeds 42, 43 e 44,
# setup de referência (seção 5) com D = 0.
ACC_ROUND = {
    "FedAvg": [0.0899, 0.306, 0.5989, 0.5824, 0.6611, 0.6977, 0.7053, 0.7254, 0.7568,
               0.7571, 0.7578, 0.7539, 0.7685, 0.7669, 0.7545, 0.7191, 0.7112, 0.6947,
               0.677, 0.6623, 0.6143],
    "LeanFed": [0.0899, 0.2219, 0.5032, 0.5527, 0.6659, 0.6888, 0.7203, 0.7394, 0.7529,
                0.7508, 0.7602, 0.7672, 0.7612, 0.7615, 0.7556, 0.7532, 0.739, 0.7442,
                0.6849, 0.6614, 0.6067],
    "LeanFed-c": [0.0899, 0.3028, 0.5261, 0.5255, 0.6193, 0.6602, 0.696, 0.6999, 0.7282,
                  0.7347, 0.7392, 0.7426, 0.7565, 0.7528, 0.7558, 0.7607, 0.7538, 0.7628,
                  0.7479, 0.7375, 0.7261],
}
NO_BATTERY_ACC = 0.7855  # mesma configuração sem energia (run 4158…, seed 42)

# Paleta categórica de scripts/plot_run.py (cores 1 a 3, nesta ordem).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_2, MUTED, AXIS, GRID_HEX = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"


def br(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


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
    # Rótulos diretos na ponta, afastados para não colidir.
    placed = []
    for y, text, color in sorted(ends):
        y_text = max([y] + [p + 0.05 for p in placed])
        placed.append(y_text)
        ax.plot(20, y, "o", ms=5, color=color, mec="white", mew=1.2, zorder=4)
        ax.text(20.4, y_text, text, fontsize=7.5, color=INK, va="center")
    ax.set_xlim(0, 24.5)
    ax.set_ylim(0, 0.9)
    ax.set_xticks(range(0, 21, 5))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.set_xlabel("rodada", fontsize=8, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID_HEX, lw=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(0.80, 0.02), frameon=False, fontsize=8,
              labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- conteúdo
def build() -> list:
    s = []
    s += [
        Spacer(1, 30),
        P("Histórico do Projeto AprendFed", "title"),
        P("O que foi implementado e testado, em ordem, e por que cada decisão foi tomada",
          "sub"),
        P("15 a 21 de outubro de 2026 — branch <i>leanfed-baseline</i> (commit 3eba2db)",
          "sub"),
        Spacer(1, 12),
        note("<b>Resumo.</b> O projeto montou um FedAvg com Flower em containers Docker, "
             "acrescentou bateria simulada com custo de energia calibrado na GPU e "
             "reproduziu o LeanFed. O LeanFed original <b>não melhora</b> sobre o FedAvg "
             "no nosso cenário, porque ignora o custo de comunicação. A versão corrigida "
             "(<b>LeanFed-c</b>), que reserva a comunicação e o custo fixo antes de "
             "escolher quantos dados treinar, chega a <b>73,7 ± 5,9%</b> de acurácia nas "
             "últimas rodadas contra <b>65,1 ± 6,0%</b> do FedAvg (3 seeds) e quase "
             "elimina os clientes que morrem por falta de bateria. Com um custo fixo de "
             "5 J por rodada, a vantagem sobe para cerca de 10 pontos. No FEMNIST, um "
             "segundo dataset com usuários reais, a vantagem some (partição por escritor) "
             "ou cai a 1,6 ponto (Dirichlet): o ganho acompanha <b>quanto das classes "
             "está nos clientes que morrem</b>, e não o simples fato de mantê-los vivos."),
        P("<b>Conteúdo</b>"),
        bullets([
            "1. Base: FedAvg com Flower e Docker (15 a 18/10)",
            "2. Escolha do modelo (15 a 18/10)",
            "3. Plano de pesquisa e modelo de energia (18 e 19/10)",
            "4. Reprodutibilidade (18 e 19/10)",
            "5. Energia e bateria simulada (20/10, manhã)",
            "6. LeanFed e LeanFed-c (20/10, tarde)",
            "7. Ajustes da reserva de comunicação (20/10, fim da tarde)",
            "8. Fração mínima de dados (20/10, noite)",
            "9. Validação com três seeds (21/10)",
            "10. Custo fixo e fração mínima 0,5 (21/10)",
            "11. Por que a seed 43 rende menos (21/10, tarde)",
            "12. Margem de cauda (21/10, tarde)",
            "13. Segundo dataset: FEMNIST (21/10, tarde)",
            "14. Controle: FEMNIST com Dirichlet (21/10, fim da tarde)",
            "15. Conclusões, limitações e próximos passos",
            "Apêndice A. Runs de referência e como reproduzir",
        ]),
    ]

    # 1 ----------------------------------------------------------------
    s += [
        H1("1. Base: FedAvg com Flower e Docker (15 a 18/10)"),
        P("<b>O que foi feito.</b> Um app Flower (<i>fedavg_app</i>) com servidor FedAvg e "
          "clientes PyTorch, cada cliente num container próprio (<i>supernode</i>) e o "
          "servidor no <i>superlink</i>, todos na mesma GPU. Os dados são o CIFAR-10, "
          "divididos entre os clientes de forma IID ou por Dirichlet (não-IID). Cada "
          "cliente separa 80% da sua partição para treino e 20% para validação; o "
          "servidor avalia o modelo global no conjunto de teste completo (10 mil imagens) "
          "a cada rodada. Os resultados vão para <i>outputs/run-&lt;id&gt;/</i>: "
          "<i>metrics.json</i> com as métricas do servidor e um <i>.jsonl</i> por cliente "
          "com cada rodada."),
        P("Também entraram as políticas de <b>clientes lentos</b> (<i>straggler-policy</i>): "
          "a velocidade de cada cliente é emulada (<i>compute-speed</i>) e o servidor pode "
          "esperar todos (<i>wait</i>), aceitar o que o cliente treinou até o prazo "
          "(<i>partial</i>) ou descartá-lo (<i>drop</i>). No dia 18, <i>run.sh</i> ganhou "
          "<i>-n</i> (número de clientes) e <i>-s</i> (velocidades), que regeneram o "
          "<i>docker-compose.yml</i>."),
        P("<b>Por quê.</b> Rodar cada cliente como um processo separado, falando com o "
          "servidor pela rede como num deploy real, evita que o simulador esconda "
          "detalhes de comunicação e de falhas. A heterogeneidade de hardware (clientes "
          "lentos) era o primeiro eixo de heterogeneidade a explorar; os testes iniciais, "
          "com 3 clientes e um deles 4× mais lento, só serviram para validar o mecanismo e "
          "não entram nos resultados."),
        P("<b>Por que o CIFAR-10.</b> O dataset veio com o app inicial, sem justificativa "
          "registrada na época; o plano de pesquisa de 18/10 (seção 3) o confirmou por dois "
          "motivos. Primeiro, é o dataset do LeanFed (CIFAR-10/100 com ResNet-18), o "
          "baseline principal, o que permite comparar com o artigo. Segundo, aceita a "
          "partição Dirichlet, cujo α controla o grau de não-IID, e a pergunta de pesquisa "
          "inclui saber se o melhor ajuste muda com esse grau. A limitação é que as imagens "
          "são de objetos genéricos e a divisão entre clientes é artificial; o mesmo plano "
          "sugere o FEMNIST, com caracteres escritos à mão por usuários reais, como segundo "
          "dataset."),
    ]

    # 2 ----------------------------------------------------------------
    s += [
        H1("2. Escolha do modelo (15 a 18/10)"),
        P("O app aceita a CNN pequena do exemplo do Flower e a MobileNetV3-Small do "
          "torchvision. Com 3 clientes, dados IID e 5 rodadas:"),
        table(
            ["Modelo", "Entrada", "Pesos iniciais", "Acurácia (5 rodadas)"],
            [
                ["CNN do projeto", "32 px", "aleatórios", "64,1%"],
                ["MobileNetV3-Small", "32 px", "aleatórios", "50,6%"],
                ["MobileNetV3-Small", "32 px", "ImageNet", "58,4%"],
                ["MobileNetV3-Small", "96 px", "ImageNet", "<b>88,6%</b>"],
            ],
            [2.2, 1, 1.4, 1.6],
        ),
        P("Com 5 clientes e 10 rodadas, a diferença se mantém: 69,1% para a CNN e 89,5% "
          "para a MobileNet pré-treinada a 96 px. Com 10 clientes, Dirichlet α = 0,1 e "
          "20 rodadas, a MobileNet chega a 82,2%."),
        P("<b>Decisão:</b> MobileNetV3-Small pré-treinada, com as imagens ampliadas para "
          "96 px. <b>Por quê:</b> é um modelo pensado para celulares, que é o tipo de "
          "dispositivo com bateria que o trabalho quer representar, e chega a uma "
          "acurácia alta em poucas rodadas. Isso deixa espaço para ver a perda causada "
          "pela bateria; com a CNN, a acurácia já era baixa sem nenhuma restrição. A "
          "ampliação para 96 px é necessária porque os pesos do ImageNet esperam imagens "
          "maiores que os 32 px do CIFAR-10 (a 32 px, os pesos pré-treinados ajudam "
          "pouco). A CNN continua disponível para testes rápidos."),
    ]

    # 3 ----------------------------------------------------------------
    s += [
        H1("3. Plano de pesquisa e modelo de energia (18 e 19/10)"),
        P("Dois documentos, nesta mesma pasta, definiram o rumo:"),
        bullets([
            "<b>heterogeneidade_bateria_fl.pdf</b> (18/10): revisão de trabalhos que "
            "juntam modelos heterogêneos e bateria, leitura detalhada do LeanFed e um "
            "roteiro em passos. O Passo 1 é ter bateria simulada no cliente e os baselines "
            "FedAvg e LeanFed; o Passo 2, submodelos de largura variável; o Passo 3, uma "
            "política que escolhe a largura conforme a bateria.",
            "<b>custo_energia_rodada.pdf</b> (18/10, revisado em 19/10): um modelo de "
            "energia por rodada com três termos, computação + comunicação + custo fixo "
            "(E = A·f(p)·η·L + C·s(p) + D), escrito na forma potência × tempo do FedLoad "
            "e calibrável com medições da GPU.",
        ]),
        P("<b>Por quê.</b> O LeanFed desconta da bateria só o custo do treino, "
          "proporcional ao volume de dados, sem unidade física e sem comunicação. Esse "
          "modelo não serve para comparar formas diferentes de economizar energia "
          "(reduzir dados, épocas ou largura do modelo), que é a pergunta da dissertação. "
          "Tudo o que foi feito depois deste ponto é o <b>Passo 1</b> do roteiro, com o "
          "modelo de energia do segundo documento."),
    ]

    # 4 ----------------------------------------------------------------
    s += [
        H1("4. Reprodutibilidade (18 e 19/10)"),
        P("<b>O que foi feito.</b> A configuração <i>seed</i> passou a fixar a divisão dos "
          "dados, os pesos iniciais, o sorteio de clientes e o embaralhamento dos batches "
          "de cada cliente (por seed, rodada e partição). O servidor passou a agregar as "
          "respostas na ordem das partições (<i>OrderedFedAvg</i>), e não na ordem de "
          "chegada. As pastas de saída ganharam a data (<i>run-&lt;id&gt;-DDMMAAAA</i>) e "
          "os containers, o fuso de São Paulo."),
        P("<b>Por quê.</b> Para comparar políticas de energia, a única diferença entre "
          "duas runs deve ser a política. A ordem de agregação importa porque somas em "
          "ponto flutuante dependem da ordem: quatro runs de teste iguais da CNN, feitas "
          "durante a mudança, deram entre 39,85% e 39,97%. Com a agregação ordenada, "
          "runs repetidas passaram a dar resultados idênticos: duas da MobileNet em 18/10 "
          "(82,41%), duas sem energia com 15 clientes em 20/10 (78,55%) e as runs 1361… "
          "e 1588… do LeanFed-c (78,37%)."),
        note("<b>Consequência usada no dia 21.</b> Como a mesma configuração com a mesma "
             "seed dá o mesmo resultado, as runs da seed 42 do dia 20 foram reaproveitadas "
             "na validação com várias seeds, e só as seeds 43 e 44 precisaram ser rodadas."),
    ]

    # 5 ----------------------------------------------------------------
    s += [
        H1("5. Energia e bateria simulada (20/10, manhã)"),
        H2("5.1 Perfil de energia da GPU"),
        P("<b>O que foi feito.</b> <i>profile_energy.py</i> mede, com o contador de energia "
          "da NVML, a energia e o tempo por amostra de treino usando a própria função "
          "<i>train()</i> dos clientes, e desconta a potência da GPU em repouso. O "
          "resultado fica em <i>fedavg_app/profiles/</i> e vai para os clientes junto "
          "com o código. Para a MobileNetV3-Small a 96 px, batch 64, numa RTX 5070 Ti:"),
        table(
            ["Grandeza", "Valor"],
            [
                ["Tempo por amostra", "93,7 µs"],
                ["Potência de treino (acima do repouso P8, 15,2 W)", "94,8 W"],
                ["Energia por amostra", "8,88 mJ"],
                ["Parâmetros do modelo", "1.528.106 (48,9 Mbit em float32)"],
            ],
            [3, 2],
        ),
        P("<b>Por quê.</b> A energia de cada rodada é <b>calculada</b> a partir do perfil "
          "(potência × tempo), e não medida durante o experimento. Medir ao vivo com 15 "
          "clientes dividindo a mesma GPU misturaria o consumo de todos e daria "
          "resultados diferentes a cada run. Com o perfil, a energia só depende do que o "
          "cliente processa e as runs continuam reproduzíveis. A linha de base é o "
          "repouso (<i>idle</i>, P8); a alternativa <i>awake</i> (GPU acordada, P1) dá "
          "cerca de 34% menos energia por amostra e está disponível, mas runs com bases "
          "diferentes não podem ser comparadas."),
        H2("5.2 Custo de uma rodada"),
        P("Cada rodada desconta, nesta ordem: o custo fixo D, o download do modelo, a "
          "computação (batch a batch) e o upload. A comunicação é potência do rádio "
          "(1,8 W) × bits / vazão, com a vazão de cada sentido sorteada em U(1, T) Mbps "
          "a cada rodada, com seed, como no FedLoad. Com T = 25 Mbps, a comunicação custa "
          "em média <b>23,6 J por rodada</b>, mas tem cauda longa: uma rodada com vazão "
          "perto de 1 Mbps custa quase 100 J."),
        P("Quando a bateria acaba, o cliente responde com o erro <i>BATTERY_DEPLETED</i> "
          "(101). Com <i>battery-empty-policy = drop</i> (usado em todos os resultados), "
          "ele morre no meio da rodada, como um FedAvg sem controle de energia; com "
          "<i>partial</i>, para a tempo de enviar o que treinou."),
        H2("5.3 O setup de referência"),
        P("Todos os resultados das seções 6 a 10 usam: 15 clientes, baterias de "
          "1500/1000/500 J (cada valor em 5 clientes), rede de 25 Mbps para todos, "
          "MobileNetV3-Small pré-treinada a 96 px, Dirichlet α = 0,1, 20 rodadas, uma "
          "época local, lr 0,01, todos os clientes sorteados em toda rodada."),
        P("<b>Por que esses números.</b> Com dados muito não-IID (α = 0,1), cada cliente "
          "tem poucas classes, então perder clientes faz o modelo esquecer classes. As "
          "baterias foram escolhidas para que parte dos clientes não aguente as 20 "
          "rodadas treinando tudo. Exemplo: o cliente 2 tem 4.816 amostras e bateria de "
          "500 J. Treinar tudo custa 4.816 × 8,88 mJ = 42,8 J; somando 23,6 J de "
          "comunicação, são cerca de 66 J por rodada, ou 1.330 J para 20 rodadas. No "
          "FedAvg, ele morre na rodada 7."),
        table(
            ["Run (seed 42)", "Acurácia final", "Clientes sem bateria no fim"],
            [
                ["FedAvg sem energia", "78,6%", "—"],
                ["FedAvg com bateria", "63,2%", "9 de 15"],
            ],
            [2.4, 1.4, 2],
        ),
        P("A bateria custa 15 pontos de acurácia ao FedAvg. Esse é o problema que as "
          "políticas de energia precisam resolver. <i>plot_run.py</i> (gráficos de "
          "acurácia, participação e bateria de cada cliente) foi criado nesse momento "
          "para acompanhar as runs."),
    ]

    # 6 ----------------------------------------------------------------
    s += [
        H1("6. LeanFed e LeanFed-c (20/10, tarde)"),
        P("<b>O que foi feito.</b> A configuração <i>energy-policy</i> define que fração η "
          "dos dados locais o cliente treina a cada rodada. O orçamento da rodada é a "
          "bateria atual dividida pelas participações que ainda faltam. Sendo A a energia "
          "de treinar todos os dados locais, C a comunicação estimada e D o custo fixo:"),
        table(
            ["Política", "Fração η", "Ideia"],
            [
                ["<i>none</i>", "1", "FedAvg: treina tudo até a bateria acabar"],
                ["<i>leanfed</i>", "orçamento / A",
                 "LeanFed original: supõe que só o treino gasta energia"],
                ["<i>leanfed-c</i>", "(orçamento − C − D) / A",
                 "LeanFed corrigido: reserva antes a rede e o custo fixo"],
            ],
            [1.1, 1.6, 3],
        ),
        P("Com η = 0, o cliente pula a rodada <b>antes de baixar o modelo</b> (erro "
          "<i>ENERGY_SKIPPED</i>, 102) e não gasta nada; a energia guardada aumenta o "
          "orçamento das rodadas seguintes."),
        P("<b>Por que corrigir o LeanFed.</b> Aqui a comunicação é de um terço a metade "
          "do custo de uma rodada, conforme o volume de dados do cliente, e ela não "
          "diminui quando o cliente treina menos dados. O "
          "LeanFed gasta o orçamento todo em treino e paga a rede por fora, então estoura "
          "o orçamento a cada rodada. No exemplo do cliente 2, o orçamento inicial é "
          "500 / 20 = 25 J: o LeanFed treina 58% dos dados (25 J) e ainda paga cerca de "
          "24 J de rede, e o cliente morre na rodada 11. O LeanFed-c reserva os 23,6 J "
          "da rede e treina com o que sobra."),
        table(
            ["Seed 42", "Acurácia final", "Rodadas <i>dead</i>", "Sem bateria no fim"],
            [
                ["FedAvg", "63,2%", "49", "9"],
                ["LeanFed", "63,6%", "32", "9"],
                ["LeanFed-c", "<b>78,4%</b>", "<b>0</b>", "1"],
            ],
            [1.8, 1.3, 1.3, 1.6],
        ),
        P("Uma rodada <i>dead</i> é uma rodada em que o servidor chamou um cliente que já "
          "estava sem bateria. <i>compare_runs.py</i>, que sobrepõe de 2 a 4 runs nos "
          "mesmos gráficos e avisa quando elas diferem em mais do que o que se testa, "
          "foi criado para essa comparação."),
    ]

    # 7 ----------------------------------------------------------------
    s += [
        H1("7. Ajustes da reserva de comunicação (20/10, fim da tarde)"),
        P("<b>O que foi feito.</b> Duas opções para a estimativa de C do LeanFed-c:"),
        bullets([
            "<i>comm-estimate</i>: reservar o custo médio (<i>mean</i>, padrão) ou um "
            "percentil dele. Com T = 25 Mbps, o p50 reserva 13,5 J, a média 23,6 J, o p75 "
            "25,1 J e o p90 51,8 J.",
            "<i>eta-after-download</i>: recalcular η depois do download, já com o custo "
            "real dele, estimando só o upload.",
        ]),
        P("<b>Por quê.</b> A vazão é sorteada a cada rodada e a média não protege contra "
          "uma rodada de rede ruim. A hipótese era que reservar mais (ou corrigir depois "
          "do download) evitaria as mortes restantes."),
        table(
            ["LeanFed-c, seed 42", "Acurácia final", "Participações", "Sem bateria no fim"],
            [
                ["média (padrão)", "78,4%", "287", "1"],
                ["p50", "73,5%", "278", "4"],
                ["p75", "74,9%", "285", "1"],
                ["p90", "78,7%", "240", "0"],
                ["média + <i>eta-after-download</i>", "78,5%", "268", "2"],
            ],
            [2.2, 1.2, 1.2, 1.5],
        ),
        P("<b>Decisão:</b> manter a <b>média</b>. Nenhuma alternativa ganhou de forma "
          "clara: o p90 evita a última morte, mas reserva tanto que os clientes "
          "participam 47 vezes a menos; o p50 reserva pouco e volta a perder clientes. "
          "Como diferenças de 3 a 4 pontos aparecem entre runs de uma seed só, não "
          "havia motivo para trocar o padrão mais simples."),
    ]

    # 8 ----------------------------------------------------------------
    s += [
        H1("8. Fração mínima de dados (20/10, noite)"),
        P("<b>Problema observado.</b> Com bateria pequena, o LeanFed-c participa de quase "
          "todas as rodadas, mas treina quase nada. O cliente 2 da seed 42 tem orçamento "
          "de 25 J e reserva 23,6 J para a rede, então η = (25 − 23,6) / 42,8 ≈ 3%: na "
          "rodada 1 ele treinou 156 amostras e gastou 13,3 J, a maior parte na rede. "
          "Na rodada 7, uma vazão ruim custou 104,5 J para treinar 253 amostras."),
        P("<b>O que foi feito.</b> <i>min-data-fraction</i>: se η sair abaixo desse "
          "valor, o cliente pula a rodada e guarda a energia para participar depois com "
          "mais dados. Na última participação esperada ele participa com qualquer η > 0, "
          "porque não há rodada seguinte para usar a energia guardada. O padrão 0 mantém "
          "o comportamento anterior."),
        P("<b>Resultado (seed 42).</b> Com 0,1, 0,2 e 0,3, a acurácia final foi 77,1%, "
          "78,1% e 77,8%, contra 78,4% sem o mínimo. Com 0,3, os cinco clientes de 500 J "
          "treinam o dobro de amostras no total (79.744 contra 39.183) em menos "
          "participações (77 contra 88): o cliente 2 pula as rodadas 1 a 7 e depois "
          "treina de 35% a 75% dos dados. Mas a acurácia não muda além do ruído de uma "
          "seed. A conclusão do dia foi que faltava validar com várias seeds antes de "
          "decidir qualquer coisa."),
    ]

    # 9 ----------------------------------------------------------------
    s += [
        H1("9. Validação com três seeds (21/10)"),
        P("<b>O que foi feito.</b> <i>scripts/run_seeds.sh</i> roda um lote de variantes "
          "em várias seeds, uma run depois da outra, com log por run em "
          "<i>logs/seeds/</i>. <i>scripts/seeds_summary.py</i> procura em "
          "<i>outputs/</i> as runs com o mesmo setup de uma run de referência (mesma "
          "configuração fora seed e política, mesmas baterias e redes), agrupa por "
          "variante e calcula média ± desvio amostral. Foram 8 runs novas (seeds 43 e "
          "44), reaproveitando a seed 42 (seção 4)."),
        P("<b>Por quê.</b> Com uma seed só, a diferença entre variantes do LeanFed-c "
          "(3 a 4 pontos) estava dentro do ruído entre runs. Para usar no artigo, as "
          "diferenças precisam aparecer em várias divisões de dados e inicializações."),
        H2("9.1 A acurácia da última rodada engana"),
        P("A primeira tabela usava a acurácia da última rodada. Ela se mostrou instável: "
          "com dados não-IID, depende de quais clientes ainda estão vivos naquela "
          "rodada específica (no teste da seção 10, o FedAvg chegou a \"melhorar\" 4,6 "
          "pontos com um custo a mais). A partir daí, a métrica de comparação passou a "
          "ser a <b>média da acurácia nas 3 últimas rodadas</b> (coluna <i>últimas</i> do "
          "<i>seeds_summary.py</i>, opção <i>--last</i>)."),
        table(
            ["Variante (3 seeds)", "Média das 3 últimas", "Acurácia máxima",
             "Rodadas <i>dead</i>", "Sem bateria no fim"],
            [
                ["FedAvg", "65,1 ± 6,0%", "77,4 ± 0,4%", "45,3 ± 4,0", "8,0 ± 1,0"],
                ["LeanFed", "65,1 ± 6,2%", "77,1 ± 2,2%", "29,0 ± 4,4", "8,0 ± 1,0"],
                ["LeanFed-c", "<b>73,7 ± 5,9%</b>", "77,1 ± 4,2%", "1,0 ± 1,7",
                 "1,3 ± 1,5"],
                ["LeanFed-c, mín. 0,3", "<b>75,7 ± 3,5%</b>", "78,5 ± 3,0%", "0,3 ± 0,6",
                 "1,0 ± 1,0"],
            ],
            [1.9, 1.5, 1.4, 1.2, 1.3],
        ),
        figure(TMP / "acuracia.png", 6.3,
               "Figura 1. Acurácia no teste por rodada, média das seeds 42, 43 e 44. As três "
               "políticas chegam ao mesmo pico (cerca de 77%); a diferença está no fim, "
               "quando os clientes de bateria pequena morrem no FedAvg e no LeanFed. O "
               "LeanFed-c começa mais devagar, porque os clientes treinam menos dados por "
               "rodada."),
        H2("9.2 O que os números dizem"),
        bullets([
            "<b>O LeanFed-c ganha em todas as seeds</b>: +10,8, +2,0 e +12,9 pontos sobre o "
            "FedAvg (média das 3 últimas rodadas). A seed 42, usada no dia 20, era a mais "
            "favorável: o ganho médio é menor do que ela mostrava.",
            "<b>A acurácia máxima é a mesma</b> nas três políticas. O ganho do LeanFed-c é "
            "não perder acurácia no fim, quando os clientes morrem.",
            "<b>O LeanFed original não melhora nada</b> sobre o FedAvg: reduz as rodadas "
            "<i>dead</i>, mas termina com os mesmos 8 clientes sem bateria.",
            "<b>O LeanFed-c nem sempre zera as mortes</b>: na seed 43 houve 3 rodadas "
            "<i>dead</i>. A reserva pela média não cobre uma sequência de rodadas de rede "
            "ruim.",
            "<b>O mínimo de 0,3</b> dá +2 pontos e desvio menor, mas o ganho aparece em 2 "
            "das 3 seeds (−0,8, +3,5 e +3,2). Com n = 3, não dá para afirmar.",
            "A energia total é igual em todas as variantes (cerca de 11,2 kJ): as baterias "
            "se esgotam de qualquer jeito, e o que muda é como a energia se distribui "
            "entre as rodadas.",
        ]),
    ]

    # 10 ---------------------------------------------------------------
    s += [
        H1("10. Custo fixo e fração mínima 0,5 (21/10)"),
        P("<b>O que foi feito.</b> <i>run_seeds.sh</i> ganhou <i>-v</i> (escolher as "
          "variantes) e <i>-x</i> (somar uma configuração a todas as runs). Foram 12 runs: "
          "FedAvg, LeanFed e LeanFed-c com <b>D = 5 J</b>, e LeanFed-c com mínimo de "
          "<b>0,5</b>, cada um nas seeds 42, 43 e 44."),
        P("<b>Por que D = 5 J.</b> No exemplo normalizado do documento de custo de "
          "energia, D = 0,1 e C = 0,5, ou seja, D ≈ C/5. Com C = 23,6 J, isso dá cerca "
          "de 5 J. Até aqui, todas as runs tinham D = 0, o que favorecia o LeanFed "
          "original (que ignora D). Como D muda o custo de todas as políticas, os "
          "baselines também foram rodados com D."),
        table(
            ["Variante (3 seeds)", "D = 0", "D = 5 J", "Rodadas <i>dead</i> (D = 0 → 5)"],
            [
                ["FedAvg", "65,1 ± 6,0%", "63,3 ± 7,1%", "45 → 56"],
                ["LeanFed", "65,1 ± 6,2%", "63,5 ± 7,3%", "29 → 41"],
                ["LeanFed-c", "73,7 ± 5,9%", "73,3 ± 5,3%", "1,0 → 0,3"],
            ],
            [1.8, 1.4, 1.4, 1.8],
        ),
        bullets([
            "<b>O LeanFed-c não é afetado por D</b>, porque já o reserva antes de escolher "
            "η. FedAvg e LeanFed perdem cerca de 1,8 ponto, e o LeanFed morre mais (a "
            "falha de ignorar D). A vantagem do LeanFed-c sobe de cerca de 8,6 para cerca "
            "de 10 pontos.",
            "<b>Mínimo 0,5:</b> 75,7 ± 3,3%, igual ao 0,3 (75,7 ± 3,5%), com menos "
            "participações (264 contra 274). Não há ganho em subir de 0,3 para 0,5.",
        ]),
    ]

    # 11 ---------------------------------------------------------------
    s += [
        H1("11. Por que a seed 43 rende menos (21/10, tarde)"),
        P("<b>Pergunta.</b> Na seed 43, o LeanFed-c ganha só 2 pontos sobre o FedAvg, contra "
          "11 a 13 nas outras, e é a única seed com rodadas <i>dead</i>. A análise usou só o "
          "que as runs gravaram (os <i>.jsonl</i> de cada cliente, a partição recriada pela "
          "seed e o <i>final_model.pt</i> avaliado por classe); não precisou de treino novo."),
        P("<b>O que aconteceu.</b> Três clientes morreram nas rodadas 18 a 20, cada um por "
          "<b>uma única rodada de rede ruim</b>, quando a bateria já não tinha folga:"),
        table(
            ["Cliente", "Bateria", "Amostras", "Rodada", "Vazão (Mbps)", "Rede gasta"],
            [
                ["5", "500 J", "6.826", "18", "9,3 / 1,2", "81,6 J"],
                ["6", "1500 J", "7.193", "19", "1,1 / 1,8", "127,2 J"],
                ["14", "500 J", "2.564", "20", "1,3 / 23,8", "52 J (todo o resto)"],
            ],
            [0.8, 0.8, 0.9, 0.8, 1.2, 1.5],
        ),
        P("A reserva é de 23,6 J (a média). No meio do treino, uma rodada assim é absorvida "
          "(o cliente 6 pagou 66 J de rede na rodada 16 e seguiu), mas o orçamento "
          "\"bateria / participações restantes\" leva a bateria a zero na última rodada, e "
          "perto do fim não há folga. Os clientes 5 e 6 são os dois maiores da seed e tinham "
          "94% dos pássaros e 92% dos cervos: no modelo final, pássaro caiu a 48% e cervo a "
          "18%. O mesmo mecanismo aparece na seed 42 (o cliente 4 morre na rodada 20 com 89% "
          "dos sapos, que caem a 47%)."),
        P("Há um segundo mecanismo, sem morte: um cliente de 500 J com muitos dados treina "
          "tão pouco por rodada que suas classes quase não entram no modelo (seed 44, "
          "cliente 8 com 7.287 amostras: cervo a 8%)."),
        note("<b>Conclusão.</b> A variância entre seeds vem sobretudo de quais classes estão "
             "com os clientes que morrem no fim, e não de ruído do treino."),
    ]

    # 12 ---------------------------------------------------------------
    s += [
        H1("12. Margem de cauda (21/10, tarde)"),
        P("<b>O que foi feito.</b> <i>tail-margin = q</i>: o cliente guarda fora do orçamento "
          "o percentil q do custo de comunicação mais o custo fixo (51,8 J com q = 0,9) até a "
          "última participação, que usa a bateria inteira e reserva a comunicação por esse "
          "percentil. <b>Por quê:</b> atacar a causa da seção 11 sem reservar o p90 em todas "
          "as rodadas, o que na seção 7 custou 47 participações."),
        table(
            ["LeanFed-c (3 seeds)", "Média das 3 últimas", "Rodadas <i>dead</i>",
             "Sem bateria no fim", "Participações"],
            [
                ["sem margem", "73,7 ± 5,9%", "1,0", "1,3", "289"],
                ["<i>tail-margin</i> = 0,9", "74,1 ± 5,4%", "<b>0</b>", "<b>0</b>", "282"],
            ],
            [1.7, 1.4, 1.1, 1.3, 1.1],
        ),
        P("<b>Resultado.</b> A margem zera as mortes nas três seeds, mas <b>não muda a "
          "acurácia</b>. Na seed 43, depois de uma rodada de rede ruim no fim o cliente "
          "passa a <b>pular</b> as últimas rodadas em vez de morrer (o cliente 5 fica com "
          "51 J, abaixo da margem; o cliente 6 gasta 170 J na rodada 19 e fica com 15 J), e "
          "o modelo final perde as classes dele do mesmo jeito (pássaro 46%, cervo 19%). "
          "<b>Decisão:</b> manter como opção, desligada por padrão. O problema de fundo é "
          "depender de clientes frágeis justamente nas últimas rodadas."),
    ]

    # 13 ---------------------------------------------------------------
    s += [
        H1("13. Segundo dataset: FEMNIST (21/10, tarde)"),
        P("<b>O que foi feito.</b> Opção <i>dataset = femnist</i>: 814 mil caracteres "
          "escritos à mão (28×28, tons de cinza, 62 classes) por 3.597 pessoas. Com "
          "<i>partitioning = writer</i>, cada cliente fica com os caracteres de 12 pessoas "
          "sorteadas pela seed (cerca de 2.700, a escala de um cliente do CIFAR-10); "
          "<i>iid</i> e <i>dirichlet</i> redistribuem os caracteres dessas mesmas pessoas. "
          "O teste centralizado usa 50 pessoas fixas, que nunca vão para um cliente. As "
          "imagens são replicadas nos 3 canais e ampliadas na GPU, então os modelos e o "
          "perfil de energia são os mesmos."),
        P("<b>Por quê.</b> O CIFAR-10 é o dataset do LeanFed, mas a divisão entre clientes é "
          "artificial; o plano de pesquisa sugeria o FEMNIST, em que cada cliente é um "
          "usuário real. Sem restrição de energia, o FEMNIST chega a 58% em 10 rodadas."),
        table(
            ["FEMNIST, por escritor (3 seeds)", "Média das 3 últimas", "Acurácia máxima",
             "Sem bateria no fim"],
            [
                ["FedAvg", "82,0 ± 1,7%", "83,0%", "5,7"],
                ["LeanFed", "80,4 ± 2,7%", "82,5%", "5,7"],
                ["LeanFed-c", "81,1 ± 1,9%", "82,8%", "1,0"],
            ],
            [2.2, 1.4, 1.3, 1.3],
        ),
        P("<b>Resultado.</b> O LeanFed-c mantém os clientes vivos, mas <b>não ganha "
          "acurácia</b> (+0,2, −2,6 e −0,1 ponto por seed). O FedAvg nem cai no fim. Na "
          "partição por escritor, <b>todo cliente tem as 62 classes</b>: um cliente que "
          "morre leva caligrafia e volume de dados, não classes inteiras."),
    ]

    # 14 ---------------------------------------------------------------
    s += [
        H1("14. Controle: FEMNIST com Dirichlet (21/10, fim da tarde)"),
        P("<b>Por quê.</b> Entre o CIFAR-10 e o FEMNIST por escritor mudaram duas coisas, o "
          "dataset e o tipo de partição. Para atribuir a diferença à heterogeneidade de "
          "rótulos, as mesmas pessoas do FEMNIST foram divididas por Dirichlet α = 0,1, "
          "como no CIFAR-10."),
        P("<b>Resultado.</b> FedAvg 69,4 ± 2,1%, LeanFed 67,7 ± 4,2% e LeanFed-c "
          "71,0 ± 3,1%: o ganho volta, mas pequeno (+0,3, +3,9 e +0,6 ponto). Para "
          "entender, foi medido quanto de cada classe estava nos clientes que o FedAvg "
          "perde antes da janela das 3 últimas rodadas:"),
        table(
            ["Cenário (3 seeds)", "Ganho do LeanFed-c", "Dados perdidos pelo FedAvg",
             "Classes com &gt;80% dos dados perdidos"],
            [
                ["CIFAR-10, Dirichlet", "<b>+8,6</b>", "52–61%", "30–40%"],
                ["FEMNIST, Dirichlet", "<b>+1,6</b>", "34–54%", "18–23%"],
                ["FEMNIST, por escritor", "<b>~0</b>", "31–33%",
                 "0% (a pior classe perde 41%)"],
            ],
            [1.7, 1.2, 1.5, 1.8],
        ),
        note("<b>Conclusão.</b> Perder volume de dados não custa acurácia: na partição por "
             "escritor, o FedAvg perde um terço dos dados e empata. Perder classes custa: o "
             "ganho do LeanFed-c acompanha a concentração de classes nos clientes de "
             "bateria fraca. Com a mesma partição, o FEMNIST tem metade das classes muito "
             "concentradas do CIFAR-10 e, com 62 classes, cada uma pesa menos na acurácia. "
             "Com 3 seeds, a relação vale entre cenários, não seed a seed."),
    ]

    # 15 ---------------------------------------------------------------
    s += [
        H1("15. Conclusões, limitações e próximos passos"),
        H2("O que já se pode afirmar"),
        bullets([
            "No nosso cenário, com comunicação realista, o LeanFed original não é melhor "
            "que o FedAvg. A correção que reserva comunicação e custo fixo é o que faz a "
            "política funcionar.",
            "O ganho vem de manter os clientes de bateria pequena (e suas classes) até o "
            "fim do treino, e não de chegar a um pico maior.",
            "O ganho depende de quanto das classes está nos clientes de bateria fraca: "
            "cerca de 9 pontos no CIFAR-10 com Dirichlet, 1,6 no FEMNIST com Dirichlet e "
            "nenhum no FEMNIST por escritor, em que todo cliente tem todas as classes.",
            "Para comparar variantes, usar a média das últimas rodadas e várias seeds; a "
            "acurácia de uma rodada e de uma seed é ruído.",
        ]),
        H2("Limitações"),
        bullets([
            "Três seeds por variante: o suficiente para o LeanFed-c contra os baselines, "
            "não para separar as variantes do LeanFed-c entre si.",
            "Setup único: uma faixa de rede (25 Mbps), sem diferença de potência entre "
            "clientes, sem recarga de bateria. A energia vem do perfil de uma GPU de "
            "desktop, não de um celular.",
            "No FEMNIST, a partição por escritor usa 12 pessoas por cliente, o que dilui a "
            "heterogeneidade entre clientes; menos pessoas por cliente deixariam a partição "
            "natural mais desigual.",
            "As mortes no fim por uma rodada de rede ruim não têm solução ainda: a margem "
            "de cauda troca a morte por rodadas puladas.",
            "Só existe a largura completa do modelo (p = 1): o modelo de energia já tem os "
            "termos f(p) e s(p), mas os submodelos ainda não foram implementados.",
        ]),
        H2("Próximos passos"),
        bullets([
            "Medir a concentração de classes nos clientes de bateria fraca como eixo dos "
            "experimentos (por exemplo, variando α ou o número de pessoas por cliente) e "
            "verificar se o ganho a acompanha também seed a seed, com mais seeds.",
            "Fazer os clientes frágeis contribuírem mais cedo, em vez de dependerem das "
            "últimas rodadas.",
            "Variar o setup (faixas de rede diferentes, baterias, potência por cliente) "
            "para ver se a vantagem se mantém.",
            "Passo 2 do roteiro: submodelos de largura variável (HeteroFL) e comparar "
            "\"reduzir dados\" (LeanFed-c) com \"reduzir modelo\".",
        ]),
    ]

    # A ----------------------------------------------------------------
    s += [
        H1("Apêndice A. Runs de referência e como reproduzir"),
        P("Pastas em <i>outputs/</i> (prefixo do id). Todas com o setup da seção 5.3; as do "
          "FEMNIST mudam só o dataset e a partição."),
        table(
            ["Variante", "Seed 42", "Seed 43", "Seed 44"],
            [
                ["FedAvg sem energia", "4158…", "—", "—"],
                ["FedAvg", "1977…", "18006…", "10759…"],
                ["LeanFed", "16069…", "11656…", "11938…"],
                ["LeanFed-c", "10494… (= 1361…, 1588…)", "12096…", "4501…"],
                ["LeanFed-c, mín. 0,3", "7022…", "12763…", "12243…"],
                ["LeanFed-c, mín. 0,5", "57608…", "11724…", "21629…"],
                ["FedAvg, D = 5 J", "99297…", "14583…", "15387…"],
                ["LeanFed, D = 5 J", "41686…", "16970…", "12657…"],
                ["LeanFed-c, D = 5 J", "10067…", "10184…", "98610…"],
                ["LeanFed-c p50 / p75 / p90", "12286… / 9450… / 3433…", "—", "—"],
                ["LeanFed-c <i>eta-after-download</i>", "6805…", "—", "—"],
                ["LeanFed-c, mín. 0,1 / 0,2", "5883… / 7923…", "—", "—"],
                ["LeanFed-c, <i>tail-margin</i> 0,9", "17135…", "16908…", "11427…"],
                ["FEMNIST escritor: FedAvg", "10788…", "31018…", "31394…"],
                ["FEMNIST escritor: LeanFed", "80811…", "94706…", "18421…"],
                ["FEMNIST escritor: LeanFed-c", "17288…", "11376…", "72789…"],
                ["FEMNIST Dirichlet: FedAvg", "38162…", "12225…", "15641…"],
                ["FEMNIST Dirichlet: LeanFed", "91848…", "74950…", "17498…"],
                ["FEMNIST Dirichlet: LeanFed-c", "14124…", "14952…", "15300…"],
            ],
            [2.2, 2.2, 1.2, 1.2],
            split=True,
        ),
        P("Com os containers de pé (<i>./run.sh -n 15 -b 1500,1000,500 --net-mbps 25</i>):"),
        code("""
# variantes padrão (FedAvg, LeanFed, LeanFed-c, LeanFed-c mín. 0,3)
./scripts/run_seeds.sh 42 43 44
# custo fixo de 5 J e mínimo 0,5
./scripts/run_seeds.sh -v fedavg,leanfed,leanfed-c -x "energy-fixed-j=5.0" 42 43 44
./scripts/run_seeds.sh -v leanfed-c-mdf0.5 42 43 44
./scripts/run_seeds.sh -v leanfed-c-tail0.9 42 43 44
# FEMNIST (baixar uma vez antes; ver README, seção Datasets)
./scripts/run_seeds.sh -d femnist -v fedavg,leanfed,leanfed-c 42 43 44
./scripts/run_seeds.sh -d femnist-dirichlet -v fedavg,leanfed,leanfed-c 42 43 44
# tabelas de média ± desvio (o --ref escolhe o setup)
python3 scripts/seeds_summary.py --ref 10494                                # CIFAR-10
python3 scripts/seeds_summary.py --ref 72789 -o graficos/seeds-femnist
python3 scripts/seeds_summary.py --ref 15300 -o graficos/seeds-femnist-dirichlet
"""),
    ]
    return s


def main() -> None:
    figure_accuracy(TMP / "acuracia.png")
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.8 * inch, bottomMargin=0.9 * inch,
        title="Histórico do Projeto AprendFed",
        subject="O que foi implementado e testado de 15 a 21/10/2026, e por quê",
        author="AprendFed",
    )
    doc.build(build(), onFirstPage=footer, onLaterPages=footer)
    print(f"{OUT} gerado.")


if __name__ == "__main__":
    main()
