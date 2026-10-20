#!/usr/bin/env bash
# Atalho para submeter um experimento FedAvg ao SuperLink.
# Exemplos:
#   ./run.sh
#   ./run.sh -c "num-server-rounds=20 lr=0.05 partitioning='dirichlet'"
#   ./run.sh -n 10                      # recria o compose com 10 clientes e roda
#   ./run.sh -n 15 -s 1.0,1.0,0.25 -c "num-server-rounds=20"
#   ./run.sh -n 6 -b 500,200 --net-mbps 75,25,5 -c "energy=true"
#
# -n/-s/-b/--power-factors/--net-mbps/--radio-w (opcionais, antes dos argumentos
# do flwr) chamam scripts/gen_compose.py e sobem os containers. Por padrão, min-nodes = número de clientes no compose,
# para a 1ª rodada só começar com todos conectados; sobrescreva com
# -c "min-nodes=..." se quiser derrubar clientes durante o experimento.
set -euo pipefail
cd "$(dirname "$0")"

gen_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--clients) gen_args+=(--clients "$2"); shift 2 ;;
    -s|--speeds)  gen_args+=(--speeds "$2"); shift 2 ;;
    -b|--batteries) gen_args+=(--batteries "$2"); shift 2 ;;
    --power-factors|--net-mbps|--radio-w) gen_args+=("$1" "$2"); shift 2 ;;
    *) break ;;
  esac
done

if [[ ${#gen_args[@]} -gt 0 ]]; then
  python3 scripts/gen_compose.py "${gen_args[@]}"
  docker compose up -d --remove-orphans
fi

n=$(docker compose config --services | grep -c '^supernode-')
exec docker compose run --rm runner flwr run /app docker --stream -c "min-nodes=$n" "$@"
