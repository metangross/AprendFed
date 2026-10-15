#!/usr/bin/env bash
# Atalho para submeter um experimento FedAvg ao SuperLink.
# Exemplos:
#   ./run.sh
#   ./run.sh -c "num-server-rounds=20 lr=0.05 partitioning='dirichlet'"
set -euo pipefail
cd "$(dirname "$0")"
exec docker compose run --rm runner flwr run /app docker --stream "$@"
