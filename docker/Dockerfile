# Imagem única usada pelo SuperLink (servidor), SuperNodes (clientes) e runner.
# PyTorch 2.7 + CUDA 12.8 é necessário para GPUs Blackwell (ex.: RTX 50xx, sm_120).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLWR_HOME=/flwr_home \
    TORCH_HOME=/data/torch

# tzdata: sem ele a variável TZ do compose é ignorada e o relógio fica em UTC.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "flwr==1.30.0" "pyarrow>=15"

# Conexão do CLI `flwr` com o SuperLink dentro da rede do compose.
COPY docker/flwr_home/config.toml /flwr_home/config.toml

WORKDIR /app
COPY pyproject.toml ./
COPY fedavg_app ./fedavg_app
# Instala o pacote sem dependências (torch/flwr já estão na imagem).
RUN pip install --no-deps -e .
