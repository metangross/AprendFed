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
ls outputs/run-<run_id>/   # final_model.pt, metrics.json, clients/

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

Para mudar a quantidade de clientes, gere o compose novamente e recrie os
containers (`docker compose up -d --remove-orphans`).

Parâmetros úteis do FedAvg em `pyproject.toml`:

- `fraction-train` / `fraction-evaluate`: fração dos clientes conectados
  sorteada a cada rodada (ex.: `0.5` = metade).
- `min-nodes`: quantos clientes precisam estar conectados para a rodada começar.
  Se você derrubar clientes e ficar abaixo desse valor, o servidor espera.

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
outputs/run-<run_id>/clients/partition-0.jsonl
outputs/run-<run_id>/clients/partition-1.jsonl
...
```

| Campo | Significado |
|---|---|
| `round`, `phase` | rodada e fase (`train` ou `evaluate`) |
| `partition_id`, `node_id`, `compute_speed` | identificação do cliente |
| `status` | `ok`, `partial` (parou no prazo) ou `dropped` (descartado) |
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

## Observações

- O código do app é enviado aos nós a cada `flwr run` (como um FAB), então
  alterações em `fedavg_app/` não exigem rebuild. Novas **dependências**
  exigem: adicione-as no `docker/Dockerfile` e rode `docker compose build`.
- Todos os clientes compartilham a mesma GPU. Com muitos clientes, reduza
  `batch-size` ou use `fraction-train < 1` se faltar memória.
- PyTorch ≥ 2.7 com CUDA 12.8 é necessário para GPUs da série RTX 50xx.
