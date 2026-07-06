#!/bin/bash
# Locate any Spider SFT/DPO adapter + Spider data on the HPC cluster.
# Run on the LOGIN node (no GPU):   bash scripts/spider_find_adapters.sh
#
# Goal: confirm whether a *Spider-trained SFT* adapter exists here. We need the
# SFT-only checkpoint (not the post-DPO one) to start the clean DPO re-run from.
set -uo pipefail

SCRATCH=/scratch/phalle.y
HOME_DIR=/home/phalle.y

echo "############ 1. ALL LoRA adapters (dirs with adapter_config.json) ############"
find "$SCRATCH" "$HOME_DIR" -maxdepth 5 -name adapter_config.json 2>/dev/null \
  | sed 's#/adapter_config.json##' | sort -u
echo "   ^ eyeball for any path containing 'spider' or a non-'bird' SFT dir."

echo
echo "############ 2. anything named *spider* (adapters, data, results) ############"
find "$SCRATCH" "$HOME_DIR" -maxdepth 4 -iname '*spider*' 2>/dev/null | sort

echo
echo "############ 3. HF cache hits for the published Spider DPO model ############"
find "$SCRATCH/hf_cache" -maxdepth 4 -iname '*sql-dpo*' 2>/dev/null
echo "   (note: that HF model is POST-DPO — NOT the SFT-only checkpoint we need)"

echo
echo "############ 4. Spider data on the cluster: dev.json + sqlite databases ############"
find "$SCRATCH" "$HOME_DIR" -maxdepth 6 -name dev.json -path '*spider*' 2>/dev/null
echo "first few spider .sqlite files (need these on /scratch for a GPU job):"
find "$SCRATCH" "$HOME_DIR" -maxdepth 7 -name '*.sqlite' -path '*spider*' 2>/dev/null | head -3

echo
echo "=================================================================="
echo "If section 1 shows ONLY bird_* adapters and section 2 finds no"
echo "Spider SFT adapter, then none exists on this cluster — it was"
echo "trained on AWS EC2. Options: (a) pull the SFT checkpoint from the"
echo "EC2/EBS volume if still alive, or (b) re-run Spider SFT on the"
echo "cluster from sft_data.json (clean, ~2h on H200)."
echo "=================================================================="
