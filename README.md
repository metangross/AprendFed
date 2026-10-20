# AprendFed

Repositório para praticar e aprender aprendizado federado.

Implementação do **FedAvg** com [Flower](https://flower.ai) 1.30 + PyTorch,
em que o servidor e cada cliente rodam em **containers Docker separados** e
treinam na **GPU** da máquina. Dataset: CIFAR-10, particionado entre os clientes.

## Arquitetura

```
                    docker compose (rede "flower")
┌───────────────────────────────┐
│ superlink  (servidor)         │  ← ServerApp: estratégia FedAvg +
│  :9092 Fleet API              │    avaliação centralizada (GPU)
│  :9093 Control API ◄──────────┼── runner (`flwr run`) submete o experimento
└──────────────▲────────────────┘
               │ gRPC
   ┌───────────┼────────────┐
supernode-0  supernode-1  supernode-N-1     ← ClientApp: treino local (GPU)
partição 0   partição 1   partição N-1        sobre a sua partição do CIFAR-10
```

| Arquivo | Função |
|---|---|
| `fedavg_app/task.py` | Modelo CNN, particionamento (IID ou Dirichlet), treino/teste |
| `fedavg_app/client_app.py` | ClientApp: `@app.train()` e `@app.evaluate()` |
| `fedavg_app/server_app.py` | ServerApp: `FedAvg`, avaliação global, salva modelo/métricas |
| `fedavg_app/energy.py` | Modelo de energia por rodada (computação + comunicação + custo fixo) |
| `fedavg_app/profile_energy.py` | Perfilamento de energia da GPU via NVML; grava em `fedavg_app/profiles/` |
| `pyproject.toml` | Hiperparâmetros do experimento (`[tool.flwr.app.config]`) |
| `docker/Dockerfile` | Imagem única (PyTorch 2.7.1 + CUDA 12.8 + Flower) |
| `scripts/gen_compose.py` | Gera o `docker-compose.yml` com N clientes e suas velocidades |
| `run.sh` | Submete um experimento |

## Pré-requisitos

- Docker com Compose v2
- Driver NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (teste: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`)

## Uso

```bash
# 1. (opcional) escolher o número de clientes — padrão: 3
python3 scripts/gen_compose.py --clients 5

# 2. construir a imagem e subir servidor + clientes
#    (na primeira vez o serviço "dataset" baixa o CIFAR-10 para um volume)
docker compose up -d --build

# 3. rodar o experimento (logs aparecem no terminal)
./run.sh
./run.sh -c "num-server-rounds=20 local-epochs=2 lr=0.05"
./run.sh -c "partitioning='dirichlet' dirichlet-alpha=0.1"   # não-IID

# 4. resultados (uma pasta por execução)
ls outputs/run-<run_id>-<DDMMAAAA>/   # final_model.pt, metrics.json, clients/

# 5. derrubar tudo
docker compose down
```

## Controlando os clientes

Cada cliente é um container (`supernode-0`, `supernode-1`, ...):

```bash
docker compose ps                         # estado de todos os nós
docker compose logs -f supernode-1        # logs de um cliente
docker compose stop supernode-2           # simular queda de um cliente
docker compose start supernode-2          # reconectar
nvidia-smi                                # ver os processos de cada container na GPU
```

### Número de clientes

O número de clientes é um parâmetro. O jeito mais rápido é usar `-n` no
`run.sh`: ele gera o compose de novo, sobe/remove os containers e roda o
experimento:

```bash
./run.sh -n 5
./run.sh -n 10 -c "num-server-rounds=20"
./run.sh -n 15 -s 1.0,1.0,0.25        # 1/3 dos clientes 4x mais lentos
```

O mesmo resultado em passos separados:

```bash
python3 scripts/gen_compose.py --clients 10
docker compose up -d --remove-orphans
./run.sh
```

O `run.sh` conta os `supernode-*` do compose e passa `min-nodes=<N>`, então a
1ª rodada só começa quando todos os clientes estão conectados. Se você vai
derrubar clientes durante o experimento, diminua o valor:
`./run.sh -c "min-nodes=2"`.

Parâmetros úteis do FedAvg em `pyproject.toml`:

- `fraction-train` / `fraction-evaluate`: fração dos clientes conectados
  sorteada a cada rodada (ex.: `0.5` = metade).
- `min-nodes`: quantos clientes precisam estar conectados para a rodada começar
  (o `run.sh` usa o número de clientes do compose).
  Se você derrubar clientes e ficar abaixo desse valor, o servidor espera.
- `seed`: fixa a divisão dos dados, os pesos iniciais do modelo, o embaralhamento
  dos batches de cada cliente e o sorteio de clientes. Duas runs com a mesma
  configuração e a mesma seed dão resultados idênticos; para comparar
  configurações, mantenha a seed e mude só o que quer testar
  (ex.: `./run.sh -c "seed=1"`). Exceções: com `straggler-policy` `partial`/`drop`
  o corte depende de tempo real; e com `fraction-train < 1` o sorteio usa os IDs
  dos nós, que podem mudar quando os containers são recriados.

## Modelos

Escolhidos no `pyproject.toml` ou na linha de comando:

| `model` | Descrição |
|---|---|
| `cnn` (padrão) | CNN simples (~620 mil parâmetros), rápida, teto de ~76% no CIFAR-10 |
| `mobilenet_v3_small` | MobileNetV3-Small do torchvision (~1,5 milhão de parâmetros), com a última camada trocada para 10 classes |

```bash
./run.sh -c "model='mobilenet_v3_small' pretrained=true image-size=96"
```

- `pretrained=true`: o **servidor** baixa os pesos do ImageNet (cache em
  `/data/torch`) e os usa como modelo global inicial; os clientes os recebem
  na 1ª rodada, sem baixar nada.
- `image-size`: amplia as imagens 32×32 na GPU antes do modelo. Modelos feitos
  para ImageNet reduzem a resolução 32×, então em 32×32 chegam a um único pixel
  no final e os pesos pré-treinados quase não ajudam. Com 96 o resultado melhora
  muito.

Resultado de 5 rodadas (3 clientes, IID, 1 época local):

| Configuração | Acurácia (teste) | Treino por rodada (cliente) |
|---|---|---|
| `cnn` | 64,1% | 0,3 s |
| `mobilenet_v3_small` (do zero, 32 px) | 50,6% | 1,5 s |
| `mobilenet_v3_small` pré-treinada, 32 px | 58,4% | 1,5 s |
| `mobilenet_v3_small` pré-treinada, 96 px | 88,6% | 1,9 s |

## Métricas por cliente

Cada cliente grava um arquivo JSON Lines com uma linha por rodada e fase:

```
outputs/run-<run_id>-<DDMMAAAA>/clients/partition-0.jsonl
outputs/run-<run_id>-<DDMMAAAA>/clients/partition-1.jsonl
...
```

| Campo | Significado |
|---|---|
| `round`, `phase` | rodada e fase (`train` ou `evaluate`) |
| `partition_id`, `node_id`, `compute_speed` | identificação do cliente |
| `status` | `ok`, `partial` (parou no prazo ou pela bateria), `dropped` (descartado pelo prazo), `depleted` (a bateria acabou nesta rodada), `dead` (já estava sem bateria) ou `skipped` (sem energia para treinar e enviar) |
| `stop_reason` | por que o treino parou antes do fim: `deadline`, `battery` ou vazio |
| `battery_before`, `battery_after`, `energy_j` | bateria (J) antes e depois da rodada e energia gasta (com `energy=true`) |
| `energy_compute_j`, `energy_comm_j`, `energy_fixed_j` | energia da computação, da comunicação (download + upload) e o custo fixo |
| `bw_down_mbps`, `bw_up_mbps`, `net_mbps`, `power_factor` | vazões sorteadas na rodada e parâmetros do cliente |
| `energy_baseline` | linha de base usada (`idle` ou `awake`) |
| `train_loss`, `steps`, `examples_processed`, `num_examples` | treino local |
| `compute_time_s` / `sim_time_s` | tempo real na GPU / tempo emulado com a lentidão |
| `eval_loss`, `eval_acc` | avaliação do modelo global na validação local |

As mesmas linhas aparecem em `docker compose logs supernode-N`. Para analisar:

```python
import pandas as pd, glob
df = pd.concat(pd.read_json(f, lines=True) for f in glob.glob("outputs/run-*/clients/*.jsonl"))
```

O `metrics.json` do servidor guarda as médias agregadas por rodada e a
configuração usada na execução.

### Gráficos de uma run

```bash
python3 scripts/plot_run.py 14452694670766510663   # id completo ou só o começo dele
python3 scripts/plot_run.py                        # a run mais recente
python3 scripts/plot_run.py --all                  # todas as runs de outputs/
python3 scripts/plot_run.py 1445 --format pdf      # PDF ou SVG para o texto
```

Roda no host (precisa de `matplotlib`) e cria uma pasta por run, com um arquivo
por gráfico:

```
graficos/run-<id>-<data>/
├── acuracia.png           teste do modelo global e média da validação dos clientes
├── participacao.png       clientes que contribuíram para a agregação em cada rodada
├── bateria_clientes.png   todos os clientes lado a lado, na mesma escala
├── bateria/cliente-NN.png um gráfico por cliente (mesma escala do painel geral)
└── resumo.csv             por cliente: amostras, rodadas em que contribuiu,
                           bateria inicial/final, rodada em que acabou e energia
                           gasta (total, cálculo e rede)
```

Os gráficos de bateria só existem em runs com `energy=true`; sem bateria
inicial, mostram a energia acumulada. Com um id, a tabela do `resumo.csv`
também é impressa no terminal.

## Clientes heterogêneos (hardware lento)

Cada cliente tem uma velocidade relativa `compute-speed` (1.0 = normal,
0.25 = 4× mais lento), definida no `--node-config` do compose:

```bash
python3 scripts/gen_compose.py --clients 3 --speeds 1.0,1.0,0.25
docker compose up -d       # recria os clientes com as novas velocidades
```

A lentidão é **emulada**: depois de cada batch o cliente dorme o tempo extra
que um hardware mais lento levaria (`sim_time = tempo_real / compute_speed`).
O que acontece com quem demora é definido pela `straggler-policy`:

| Política | Comportamento | Efeito no treino |
|---|---|---|
| `wait` (padrão) | servidor espera todos | só atrasa a rodada; resultado igual |
| `partial` | cliente para ao atingir `round-deadline` e envia o que treinou | cliente lento faz menos passos e pesa menos no FedAvg (peso = exemplos processados) |
| `drop` | cliente que estoura `round-deadline` é descartado da rodada | seus dados não entram na agregação (FedAvg original com stragglers) |

```bash
./run.sh -c "straggler-policy='partial' round-deadline=0.5"
./run.sh -c "straggler-policy='drop' round-deadline=0.5 partitioning='dirichlet' dirichlet-alpha=0.1"
```

O prazo é em segundos de tempo emulado e depende da sua GPU: rode uma vez com
`wait` e veja `sim_time_s` nos `.jsonl` para calibrar. O efeito na acurácia
aparece mais com dados não-IID (o cliente descartado tem classes que os
outros não têm).

## Energia e bateria

O modelo de energia segue `docs/custo_energia_rodada.pdf` (seções 1 e 7),
por enquanto só com o modelo completo (largura 1):

```
E = n_amostras · P_cmp · t_amostra / v  +  P_rádio · bits · (1/vazão_down + 1/vazão_up)  +  D
```

**1. Perfilamento (uma vez por GPU e configuração).** Mede, com o NVML, a
potência em repouso e a energia e o tempo por amostra do treino, usando o mesmo
`train()` dos clientes. Como o NVML mede a placa inteira, derrube os outros
containers antes:

```bash
docker compose down
docker compose run --rm profiler            # usa model/image-size/batch-size do pyproject
docker compose run --rm profiler python -m fedavg_app.profile_energy --model mobilenet_v3_small --image-size 96
```

O resultado vai para `fedavg_app/profiles/<model>-i<image-size>-b<batch-size>.json`.
Ele deve ser versionado junto com o código, porque os clientes o recebem no FAB.
Durante os experimentos, a energia é **calculada** a partir desse perfil, e não
medida, então as runs com a mesma seed dão os mesmos níveis de bateria. O
arquivo guarda duas linhas de base: o repouso (P8, usado como `train_power_w`)
e a GPU acordada sem calcular (P1, ~15 s após a carga, em `*_awake_*`).

**2. Clientes.** Bateria inicial, potência, rede e rádio de cada cliente vão no
`--node-config`, como o `compute-speed`. As listas se repetem como em `--speeds`:

```bash
./run.sh -n 6 -s 1.0,0.5 -b 500,200 --net-mbps 75,25,5 --power-factors 1.0,0.57,0.43 \
         -c "energy=true num-server-rounds=20"
```

| Opção (`gen_compose.py` / `run.sh`) | `--node-config` | Significado |
|---|---|---|
| `-b`, `--batteries` | `battery-init` | bateria inicial (J); sem ela, a energia só é contabilizada |
| `--power-factors` | `power-factor` | potência de treino = P_ref do perfil × fator |
| `--net-mbps` | `net-mbps` | teto da vazão; a cada rodada, vazão de download e de upload sorteadas em U(1, teto) |
| `--radio-w` | `radio-w` | potência do rádio ao baixar e enviar (W) |

**3. Configuração do run** (`pyproject.toml`):

- `energy`: liga a contabilização (padrão `false`, o que não muda nada no FedAvg).
- `energy-fixed-j`: custo fixo D por rodada (J).
- `energy-baseline`: linha de base descontada da energia medida. `idle`
  (repouso, P8; padrão) cobra do treino o cálculo e o clock alto; `awake`
  (GPU acordada, P1) cobra só o cálculo e dá cerca de 36% menos energia por
  amostra. Não misture bases numa mesma comparação.
- `battery-empty-policy`: `drop` (o cliente treina até a bateria acabar e é
  descartado, como um FedAvg sem controle de energia) ou `partial` (para a tempo
  de enviar o modelo e envia o que treinou, com peso menor).

As escolhas ficam registradas na seção `energy` do `metrics.json`: base usada
e sua potência, perfil (arquivo, GPU, data), tempo e energia por amostra de
referência, custo fixo e `battery-empty-policy`. Cada linha de treino dos
`.jsonl` também traz `energy_baseline`.

Cada rodada desconta, nesta ordem, o custo fixo e o download, a computação
(batch a batch, dentro do `train()`) e o upload. Um cliente sem bateria responde
com o erro `BATTERY_DEPLETED` (101) no treino e na avaliação. A avaliação não
gasta bateria.

## Observações

- O código do app é enviado aos nós a cada `flwr run` (como um FAB), então
  alterações em `fedavg_app/` não exigem rebuild. Novas **dependências**
  exigem: adicione-as no `docker/Dockerfile` e rode `docker compose build`.
- Todos os clientes compartilham a mesma GPU. Com muitos clientes, reduza
  `batch-size` ou use `fraction-train < 1` se faltar memória.
- PyTorch ≥ 2.7 com CUDA 12.8 é necessário para GPUs da série RTX 50xx.
