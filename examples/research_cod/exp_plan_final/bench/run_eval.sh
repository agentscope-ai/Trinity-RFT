#!/usr/bin/env bash
# OOD-generalization eval: evaluate trained checkpoint(s) on a harder in-domain environment and on
# unseen cross-domain environments. Each job bench-evals EVERY saved checkpoint (a reward curve over
# checkpoint step). ckpt dir = <root>/trinity-cod-final/<train-task>/<name>; jobs use pack_size=8, 1 node x 8 GPU.
#
#   frozen_lake_obscure ckpt:  flobs (FrozenLake-hard, in-domain) + alchemy (easy) + terminal
#   mixed_flobs_alchran ckpt:  flobs (FrozenLake-hard, in-domain) + alchemy (hard) + terminal
#
# Usage (run from repo root; set TRINITY_CHECKPOINT_ROOT_DIR if ckpts aren't at the yaml default):
#   bash .../run_eval.sh --train-tasks <task...> [--eval-tasks <env...>]
#     --train-tasks   frozen_lake_obscure | mixed_flobs_alchran   (one or more)
#     --eval-tasks    flobs | alchemy | terminal                  (default: all 3; comma- or space-separated)
#
#   bash .../run_eval.sh --train-tasks frozen_lake_obscure
#   bash .../run_eval.sh --train-tasks frozen_lake_obscure mixed_flobs_alchran --eval-tasks alchemy terminal
set -u
B="examples/research_cod/exp_plan_final/bench"
SEED="${EVAL_SEED:-1}"

TRAIN_TASKS=()
EVAL_TASKS=()
mode=""
while [ $# -gt 0 ]; do
  case "$1" in
    --train-tasks|--train-task) mode=train ;;
    --eval-tasks|--eval-task)   mode=eval ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    --*) echo "Unknown flag: $1"; exit 1 ;;
    *) case "$mode" in
         train) TRAIN_TASKS+=("$1") ;;
         eval)  EVAL_TASKS+=(${1//,/ }) ;;
         *) echo "Pass --train-tasks before task names"; exit 1 ;;
       esac ;;
  esac
  shift
done
[ ${#TRAIN_TASKS[@]} -ge 1 ] || { echo "usage: run_eval.sh --train-tasks <frozen_lake_obscure|mixed_flobs_alchran>... [--eval-tasks <flobs|alchemy|terminal>...]"; exit 1; }
[ ${#EVAL_TASKS[@]} -ge 1 ] || EVAL_TASKS=(flobs alchemy terminal)

for t in "${TRAIN_TASKS[@]}"; do case "$t" in frozen_lake_obscure|mixed_flobs_alchran) ;; *) echo "unknown train-task: $t"; exit 1 ;; esac; done
for e in "${EVAL_TASKS[@]}";  do case "$e" in flobs|alchemy|terminal) ;;            *) echo "unknown eval-task: $e";  exit 1 ;; esac; done

name_of() { case "$1" in frozen_lake_obscure) echo qwen3-8b-flobs ;; mixed_flobs_alchran) echo qwen3-8b-mixed ;; esac; }
cfg_of()  { # <train-task> <eval-env> -> bench config name (alchemy difficulty follows the train task)
  case "$2" in
    flobs)    echo eval_frozenlake_obscure_hard ;;
    alchemy)  case "$1" in frozen_lake_obscure) echo eval_alchemy_easy ;; mixed_flobs_alchran) echo eval_alchemy_hard ;; esac ;;
    terminal) echo eval_terminal ;;
  esac
}

TOTAL=$(( ${#TRAIN_TASKS[@]} * ${#EVAL_TASKS[@]} )); DONE=0
echo "############ run_eval: train=[${TRAIN_TASKS[*]}]  eval=[${EVAL_TASKS[*]}]  ->  $TOTAL jobs ############"
for t in "${TRAIN_TASKS[@]}"; do
  NAME=$(name_of "$t")
  for e in "${EVAL_TASKS[@]}"; do
    cfg=$(cfg_of "$t" "$e"); DONE=$((DONE + 1))
    echo ">>> [$DONE/$TOTAL] train=$t name=$NAME eval=$e seed=$SEED cfg=$cfg"
    EVAL_GROUP="$t" EVAL_NAME="$NAME" EVAL_SEED="$SEED" trinity run --config "$B/$cfg.yaml" || echo "  !!! FAILED: train=$t eval=$e"
  done
done
echo "############ done: $DONE/$TOTAL ############"
