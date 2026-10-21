#!/usr/bin/env bash
# Roda FedAvg, LeanFed, LeanFed-c e LeanFed-c com min-data-fraction=0.3 em
# várias seeds, um run depois do outro, com o compose atual (15 clientes,
# baterias 1500/1000/500 J, 25 Mbps). Depois: python3 scripts/seeds_summary.py
#
# Exemplos:
#   ./scripts/run_seeds.sh              # seeds 43 e 44 (a 42 já está em outputs/)
#   ./scripts/run_seeds.sh 42 43 44
#   nohup ./scripts/run_seeds.sh > /dev/null 2>&1 &   # sobrevive ao terminal
#
# O log de cada run vai para logs/seeds/. Um run que falha é anotado e
# o lote continua.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ $# -gt 0 ]]; then seeds=("$@"); else seeds=(43 44); fi

base="num-server-rounds=20 model='mobilenet_v3_small' pretrained=true image-size=96"
base+=" partitioning='dirichlet' dirichlet-alpha=0.1 energy=true"
variants=(
  "fedavg|energy-policy='none'"
  "leanfed|energy-policy='leanfed'"
  "leanfed-c|energy-policy='leanfed-c'"
  "leanfed-c-mdf0.3|energy-policy='leanfed-c' min-data-fraction=0.3"
)

# Fora de outputs/, que pertence ao root dos containers.
logs=logs/seeds
mkdir -p "$logs" || exit 1
summary="$logs/lote-$(date +%Y%m%d-%H%M).txt"
total=$(( ${#seeds[@]} * ${#variants[@]} ))
i=0
for seed in "${seeds[@]}"; do
  for v in "${variants[@]}"; do
    name=${v%%|*}
    cfg=${v#*|}
    i=$((i + 1))
    log="$logs/$name-seed$seed-$(date +%H%M%S).log"
    echo "[$i/$total] $(date +%H:%M) $name seed=$seed -> $log" | tee -a "$summary"
    if ./run.sh -c "$base $cfg seed=$seed" < /dev/null > "$log" 2>&1; then
      echo "    ok $(date +%H:%M)" | tee -a "$summary"
    else
      echo "    FALHOU (código $?), veja $log" | tee -a "$summary"
    fi
  done
done
echo "Fim do lote: $(date +%H:%M). Resumo: python3 scripts/seeds_summary.py" | tee -a "$summary"
