#!/usr/bin/env python3
"""Gera o docker-compose.yml com N clientes (SuperNodes).

Uso:
    python3 scripts/gen_compose.py --clients 5
    python3 scripts/gen_compose.py --clients 3 --speeds 1.0,1.0,0.25

Cada cliente recebe um `partition-id` diferente, de modo que treina sobre um
pedaço distinto do CIFAR-10. Todos usam a GPU da máquina.

`--speeds` define a velocidade relativa de cada cliente (1.0 = normal,
0.25 = hardware emulado 4x mais lento). Se houver menos valores que clientes,
a lista é repetida: `--speeds 1.0,1.0,0.25` deixa 1/3 dos clientes lentos
com 3, 6, 15... clientes.
"""

import argparse
from pathlib import Path

HEADER = """\
# ARQUIVO GERADO por scripts/gen_compose.py -- edite o script, não este arquivo.
# Clientes: {n} | velocidades: {speeds}

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
      - --node-config=partition-id={i} num-partitions={n} compute-speed={speed}
    depends_on: [superlink]
"""

FOOTER = """
networks:
  flower:

volumes:
  data:
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", "-n", type=int, default=3)
    parser.add_argument(
        "--speeds", "-s", default="1.0",
        help="velocidades relativas separadas por vírgula (ex.: 1.0,1.0,0.25)",
    )
    parser.add_argument("--output", "-o", default="docker-compose.yml")
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients deve ser >= 1")

    speeds = [float(v) for v in args.speeds.split(",")]
    if len(speeds) > args.clients:
        parser.error("--speeds tem mais valores do que clientes")
    speeds = [speeds[i % len(speeds)] for i in range(args.clients)]
    if any(v <= 0 for v in speeds):
        parser.error("--speeds deve conter apenas valores > 0")

    text = HEADER.format(n=args.clients, speeds=",".join(map(str, speeds)))
    text += "".join(
        CLIENT.format(i=i, n=args.clients, speed=speeds[i]) for i in range(args.clients)
    )
    text += FOOTER
    out = Path(args.output)
    out.write_text(text)
    print(f"{out} gerado com {args.clients} cliente(s).")


if __name__ == "__main__":
    main()
