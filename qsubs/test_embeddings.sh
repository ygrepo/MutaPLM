#!/bin/bash
# test_embeddings.sh — submit test_embeddings jobs to LSF GPU queue


#BSUB -J test_embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o logs/test_embeddings.%J.out
#BSUB -e logs/test_embeddings.%J.err

set -euo pipefail

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest

ml proxies/1 || true

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"

cd /sc/arion/projects/DiseaseGeneCell/Huang_lab_project/MutaPLM  # adjust if needed


LOG_DIR="logs"
LOG_LEVEL="INFO"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
export NUMEXPR_MAX_THREADS=64

/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env/bin/python \
  scripts/test_embedding.py \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL"
