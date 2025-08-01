#!/bin/bash
# create_load_model.sh — submit create_load_model jobs to LSF GPU queue


#BSUB -J create_load_model
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o logs/create_load_model.%J.out
#BSUB -e logs/create_load_model.%J.err

set -euo pipefail

module purge
module load cuda/11.8 cudnn
module load anaconda3/latest
#module load anaconda3/2024.06

ml proxies/1 || true

export PROJ=/sc/arion/projects/DiseaseGeneCell/Huang_lab_data
export CONDARC="$PROJ/conda/condarc"
LOG_DIR="logs"
LOG_LEVEL="INFO"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/create_load_model_${TIMESTAMP}.log"

#eval "$(/hpc/packages/minerva-rocky9/anaconda3/2024.06/bin/conda shell.bash hook)"
#conda activate /sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env

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

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
export NUMEXPR_MAX_THREADS=64

/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env/bin/python \
  scripts/create_load_model.py \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
2>&1 | tee -a "$LOG_FILE"

# Check the exit status of the Python script
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully at $(date)" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Error: Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    echo "Check the log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi
