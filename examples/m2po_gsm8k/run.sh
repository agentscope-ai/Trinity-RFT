#!/bin/bash
trinity run --config examples/m2po_gsm8k/trainer.yaml 2>&1 | tee m2po_trainer.log &
sleep 30
trinity run --config examples/m2po_gsm8k/explorer.yaml 2>&1 | tee m2po_explorer.log &
