#!/usr/bin/env python3
"""Gera o docker-compose.yml com N clientes (SuperNodes).

Uso:
    python3 scripts/gen_compose.py --clients 5
    python3 scripts/gen_compose.py --clients 3 --speeds 1.0,1.0,0.25
    python3 scripts/gen_compose.py -n 6 --batteries 500,200 --net-mbps 75,25,5

Cada cliente recebe um `partition-id` diferente, de modo que treina sobre um
pedaço distinto do CIFAR-10. Todos usam a GPU da máquina.

`--speeds` define a velocidade relativa de cada cliente (1.0 = normal,
0.25 = hardware emulado 4x mais lento). Se houver menos valores que clientes,
a lista é repetida: `--speeds 1.0,1.0,0.25` deixa 1/3 dos clientes lentos
com 3, 6, 15... clientes. As listas de energia (usadas com `energy=true`, ver
docs/custo_energia_rodada.pdf) seguem a mesma regra:

    --batteries      bateria inicial em J (sem ela, a energia só é contabilizada)
    --power-factors  potência de treino = P_ref medida no perfil × fator
    --net-mbps       teto da vazão de rede; a vazão da rodada é sorteada em U(1, teto)
    --radio-w        potência do rádio ao baixar e enviar o modelo (W)
"""

import argparse
from pathlib import Path

HEADER = """\
# ARQUIVO GERADO por scripts/gen_compose.py -- edite o script, não este arquivo.
# Clientes: {n} | velocidades: {speeds}
# Baterias (J): {batteries} | fatores de potência: {power} | rede (Mbps): {net} | rádio (W): {radio}

x-gpu: &gpu
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

x-base: &base
  image: aprendfed:latest
  networks: [flower]
  environment:
    TZ: America/Sao_Paulo   # data/hora usadas no nome da pasta e nos JSONs da run
  volumes:
    - data:/data
    - ./outputs:/outputs

services:
  # Baixa o CIFAR-10 uma vez para o volume compartilhado.
  dataset:
    <<: *base
    build:
      context: .
      dockerfile: docker/Dockerfile
    command: ["python", "-m", "fedavg_app.prepare_data", "/data"]
    restart: "no"

  # Servidor Flower. Com --isolation subprocess o ServerApp (FedAvg) roda
  # dentro deste container, inclusive a avaliação centralizada na GPU.
  superlink:
    <<: [*base, *gpu]
    container_name: superlink
    command:
      - flower-superlink
      - --insecure
      - --isolation=subprocess
    ports:
      - "9093:9093"   # Control API (usada pelo `flwr run`)
    depends_on:
      dataset:
        condition: service_completed_successfully

  # Executa o experimento: `docker compose run --rm runner`
  runner:
    <<: *base
    profiles: [runner]
    volumes:
      - ./:/app
    command: ["flwr", "run", "/app", "docker", "--stream"]
    depends_on: [superlink]

  # Perfilamento de energia da GPU: `docker compose down && docker compose run --rm profiler`
  # O NVML mede a placa inteira, então rode sem os outros containers. Grava em
  # fedavg_app/profiles/ (o repositório é montado em /app).
  profiler:
    <<: [*base, *gpu]
    profiles: [profiler]
    volumes:
      - data:/data
      - ./:/app
    command: ["python", "-m", "fedavg_app.profile_energy"]
    depends_on:
      dataset:
        condition: service_completed_successfully
"""

CLIENT = """
  supernode-{i}:
    <<: [*base, *gpu]
    container_name: supernode-{i}
    command:
      - flower-supernode
      - --insecure
      - --superlink=superlink:9092
      - --isolation=subprocess
      - --clientappio-api-address=0.0.0.0:9094
      - --node-config=partition-id={i} num-partitions={n} compute-speed={speed}{energy}
    depends_on: [superlink]
"""

FOOTER = """
networks:
  flower:

volumes:
  data:
"""


def per_client(parser: argparse.ArgumentParser, name: str, raw: str, n: int,
               zero: bool = False) -> list[float]:
    """Lista separada por vírgulas, repetida até ter `n` valores (todos > 0, ou ≥ 0
    com `zero`)."""
    vals = [float(v) for v in raw.split(",")]
    if len(vals) > n:
        parser.error(f"{name} tem mais valores do que clientes")
    if any(v < 0 or (v == 0 and not zero) for v in vals):
        parser.error(f"{name} deve conter apenas valores > 0" + (" ou 0" if zero else ""))
    return [vals[i % len(vals)] for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", "-n", type=int, default=3)
    parser.add_argument(
        "--speeds", "-s", default="1.0",
        help="velocidades relativas separadas por vírgula (ex.: 1.0,1.0,0.25)",
    )
    parser.add_argument("--batteries", "-b", default=None, help="bateria inicial (J)")
    parser.add_argument("--power-factors", default="1.0", help="fator sobre P_ref")
    parser.add_argument("--net-mbps", default="25", help="teto da vazão (Mbps)")
    parser.add_argument("--radio-w", default="1.8", help="potência do rádio (W)")
    parser.add_argument("--output", "-o", default="docker-compose.yml")
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients deve ser >= 1")

    n = args.clients
    speeds = per_client(parser, "--speeds", args.speeds, n)
    power = per_client(parser, "--power-factors", args.power_factors, n)
    net = per_client(parser, "--net-mbps", args.net_mbps, n)
    # Rádio 0 = comunicação de graça (isola o efeito do custo de rede).
    radio = per_client(parser, "--radio-w", args.radio_w, n, zero=True)
    batteries = per_client(parser, "--batteries", args.batteries, n) if args.batteries else None

    def energy(i: int) -> str:
        out = f" power-factor={power[i]} net-mbps={net[i]} radio-w={radio[i]}"
        return out + (f" battery-init={batteries[i]}" if batteries else "")

    def fmt(vals: list[float] | None) -> str:
        return ",".join(map(str, vals)) if vals else "sem bateria"

    text = HEADER.format(
        n=n, speeds=fmt(speeds), batteries=fmt(batteries),
        power=fmt(power), net=fmt(net), radio=fmt(radio),
    )
    text += "".join(
        CLIENT.format(i=i, n=n, speed=speeds[i], energy=energy(i)) for i in range(n)
    )
    text += FOOTER
    out = Path(args.output)
    out.write_text(text)
    print(f"{out} gerado com {n} cliente(s).")


if __name__ == "__main__":
    main()
