#!/bin/bash
# extract_embeddings.sh — submit extract_embeddings jobs to LSF GPU queue


#BSUB -J extract_embeddings
#BSUB -P acc_DiseaseGeneCell
#BSUB -q gpu
#BSUB -gpu "num=1"
#BSUB -R h100nvl
#BSUB -n 1
#BSUB -R "rusage[mem=32000]"
#BSUB -W 0:30
#BSUB -o logs/extract_embeddings.%J.out
#BSUB -e logs/extract_embeddings.%J.err

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
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/extract_embeddings_${TIMESTAMP}.log"

DATASET_DIR="data/mutadescribe_data"
DATA_FN="${DATASET_DIR}/structural_split/train.csv"
OUTPUT_DIR="output/data"
OUTPUT_FN="${OUTPUT_DIR}/structural_split_train_with_embeddings.csv"
N=2000
SEED=42

mkdir -p "$OUTPUT_DIR"


export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.8"
export NUMEXPR_MAX_THREADS=64

/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.conda/envs/mutaplm_env/bin/python \
  scripts/extract_embedding.py \
  --log_dir "$LOG_DIR" \
  --log_level "$LOG_LEVEL" \
  --data_fn "$DATA_FN" \
  --output_fn "$OUTPUT_FN" \
  --n "$N" \
  --seed "$SEED" \
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