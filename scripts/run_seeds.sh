#!/usr/bin/env bash
# Roda variantes de política de energia em várias seeds, um run depois do
# outro, com o compose atual (15 clientes, baterias 1500/1000/500 J, 25 Mbps).
# Depois: python3 scripts/seeds_summary.py
#
# Exemplos:
#   ./scripts/run_seeds.sh              # 4 variantes padrão, seeds 43 e 44
#   ./scripts/run_seeds.sh 42 43 44
#   ./scripts/run_seeds.sh -v fedavg,leanfed,leanfed-c -x "energy-fixed-j=5.0" 42 43 44
#   ./scripts/run_seeds.sh -d femnist -v fedavg,leanfed,leanfed-c 42 43 44
#   nohup ./scripts/run_seeds.sh > /dev/null 2>&1 &   # sobrevive ao terminal
#
# -v: variantes (nomes de VARIANTS, separados por vírgula; padrão DEFAULT)
# -x: config somada a todas as runs do lote (ex.: custo fixo)
# -d: dataset e partição (chaves de DATA; padrão cifar10, Dirichlet α = 0,1)
#
# O log de cada run vai para logs/seeds/. Um run que falha é anotado e
# o lote continua.
set -uo pipefail
cd "$(dirname "$0")/.."

declare -A VARIANTS=(
  [fedavg]="energy-policy='none'"
  [leanfed]="energy-policy='leanfed'"
  [leanfed-c]="energy-policy='leanfed-c'"
  [leanfed-c-mdf0.3]="energy-policy='leanfed-c' min-data-fraction=0.3"
  [leanfed-c-mdf0.5]="energy-policy='leanfed-c' min-data-fraction=0.5"
  [leanfed-c-tail0.9]="energy-policy='leanfed-c' tail-margin=0.9"
)
DEFAULT=fedavg,leanfed,leanfed-c,leanfed-c-mdf0.3
declare -A DATA=(
  [cifar10]="dataset='cifar10' partitioning='dirichlet' dirichlet-alpha=0.1"
  [femnist]="dataset='femnist' partitioning='writer'"
  [femnist-dirichlet]="dataset='femnist' partitioning='dirichlet' dirichlet-alpha=0.1"
)

names=$DEFAULT
extra=""
data=cifar10
while getopts "v:x:d:" opt; do
  case $opt in
    v) names=$OPTARG ;;
    x) extra=$OPTARG ;;
    d) data=$OPTARG ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ $# -gt 0 ]]; then seeds=("$@"); else seeds=(43 44); fi
IFS=, read -ra variants <<< "$names"
for name in "${variants[@]}"; do
  [[ -v VARIANTS[$name] ]] || { echo "variante desconhecida: $name"; exit 2; }
done
[[ -v DATA[$data] ]] || { echo "dataset desconhecido: $data"; exit 2; }

base="num-server-rounds=20 model='mobilenet_v3_small' pretrained=true image-size=96"
base+=" ${DATA[$data]} energy=true $extra"
tag=$(tr -c 'A-Za-z0-9.=\n-' '_' <<< "$extra")
[[ $data != cifar10 ]] && tag=$data${tag:+-$tag}

# Fora de outputs/, que pertence ao root dos containers.
logs=logs/seeds
mkdir -p "$logs" || exit 1
summary="$logs/lote-$(date +%Y%m%d-%H%M).txt"
[[ $data != cifar10 ]] && echo "dados: ${DATA[$data]}" | tee -a "$summary"
[[ -n $extra ]] && echo "config extra: $extra" | tee -a "$summary"
total=$(( ${#seeds[@]} * ${#variants[@]} ))
i=0
for seed in "${seeds[@]}"; do
  for name in "${variants[@]}"; do
    i=$((i + 1))
    log="$logs/$name${tag:+-$tag}-seed$seed-$(date +%H%M%S).log"
    echo "[$i/$total] $(date +%H:%M) $name seed=$seed -> $log" | tee -a "$summary"
    if ./run.sh -c "$base ${VARIANTS[$name]} seed=$seed" < /dev/null > "$log" 2>&1; then
      echo "    ok $(date +%H:%M)" | tee -a "$summary"
    else
      echo "    FALHOU (código $?), veja $log" | tee -a "$summary"
    fi
  done
done
echo "Fim do lote: $(date +%H:%M). Resumo: python3 scripts/seeds_summary.py" | tee -a "$summary"
