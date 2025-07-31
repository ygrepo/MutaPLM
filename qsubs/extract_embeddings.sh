#!/bin/bash
# submit_embeddings.sh — submit embedding jobs to LSF GPU queue


#BSUB -J embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "rusage[mem=3000]"
#BSUB -W 2:00
#BSUB -o logs/embeddings.%J.out
#BSUB -e logs/embeddings.%J.err

set -euo pipefail

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
ml proxies/1 || true

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
# export HF_HOME="$PROJ/.cache/huggingface"
# export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# export TORCH_HOME="$PROJ/.cache/torch"
# export TMPDIR="$PROJ/.tmp"
# mkdir -p logs "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TMPDIR"

#conda activate "$PROJ/.conda/envs/mutaplm_env"

# verify PyTorch is built for CUDA 11.8
# python - <<'PY'
# import torch
# print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(), "cuda:", torch.version.cuda)
# PY

cd /sc/arion/projects/DiseaseGeneCell/Huang_lab_project/MutaPLM  # adjust if needed


#source /hpc/packages/minerva-centos7/anaconda3/2023.09/etc/profile.d/conda.sh
#conda activate mutaplm
#conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env

DATASET_DIR="mutadescribe_data"
DATA_FN="${DATASET_DIR}/structural_split/train.csv"
OUTPUT_DIR="output/data"
OUTPUT_FN="${OUTPUT_DIR}/structural_split_train_with_embeddings.csv"
MODEL_NAME="facebook/esm2_t6_8M_UR50D"
N=15
LOG_DIR="logs"
LOG_LEVEL="INFO"
SEED=42

/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env/bin/python \
  scripts/extract_embeddings.py \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL"
