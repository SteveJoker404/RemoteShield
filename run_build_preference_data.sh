################################################################################
# Preference-data construction launcher
#
# This script:
#   - builds DPO training samples from aligned clean and perturbed data
#   - evaluates four image-text combinations for each sample pair
#   - draws four samples for each combination, producing 16 candidates in total
#   - selects the preferred and rejected responses from the pooled candidates
#   - supports multi-GPU execution and resume-from-checkpoint behavior
################################################################################

# =============================== Required args ===============================

# Required: clean-sample JSON file. 
CLEAN_JSON="/path/to/your/file"

# Required: perturbed-sample JSON file. 
PERT_JSON="/path/to/your/file"

# Required: image root directory used to resolve relative paths from the JSON files. For example, origin/xxx.jpg becomes /path/to/your/file/origin/xxx.jpg.
IMAGE_BASE_DIR="/path/to/your/file"

# Required: visible physical GPU ids, separated by commas.
VISIBLE_GPUS="0,1,2,3,4,5"

# Required: number of GPUs used inside the script. This is usually the same as
# the number of visible GPUs.
NUM_GPUS=6

# =============================== Optional args ===============================

# Output directory for dpo_clean.jsonl and dpo_pert.jsonl.
OUTPUT_DIR="dpo_dataset"

# Model path or model name.
MODEL_NAME="/path/to/your/file"

# Number of samples per combination. The script evaluates
# 4 combinations x NUM_SAMPLES responses.
NUM_SAMPLES=4

# Sampling temperature.
TEMPERATURE=0.7

# Maximum number of generated tokens.
MAX_NEW_TOKENS=1024

# Use this flag when the model predicts boxes in 0-1000 normalized coordinates.
# Qwen3.5 / Qwen3-VL: set "--pred_bbox_norm1000"
# Qwen2.5-VL: leave this empty.
PRED_BBOX_NORM1000="--pred_bbox_norm1000"

# Leave empty to resume automatically. Set "--no_resume" to restart from scratch.
NO_RESUME=""

# ============================== Execute script ==============================

echo "========================================================================"
echo "Preference data construction (4 combinations x ${NUM_SAMPLES} samples = $((4 * NUM_SAMPLES)) generations)"
echo "========================================================================"
echo "Clean JSON      : $CLEAN_JSON"
echo "Pert JSON       : $PERT_JSON"
echo "Image base dir  : $IMAGE_BASE_DIR"
echo "Output dir      : $OUTPUT_DIR"
echo "Model           : $MODEL_NAME"
echo "Visible GPUs    : $VISIBLE_GPUS (physical ids)"
echo "Num GPUs        : $NUM_GPUS (relative ids: 0-$((NUM_GPUS-1)))"
echo "Resume          : $([ -z "$NO_RESUME" ] && echo 'enabled' || echo 'disabled (restart from scratch)')"
echo "========================================================================"

# Check CUDA availability.
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi was not found. Please check your CUDA installation."
    exit 1
fi

echo ""
echo "Visible GPUs through CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}:"
export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Environment variables.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Build the relative GPU id list: 0,1,2,...,NUM_GPUS-1.
GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))

# Build command arguments.
CMD_ARGS=(
    --clean_json "$CLEAN_JSON"
    --pert_json "$PERT_JSON"
    --image_base_dir "$IMAGE_BASE_DIR"
    --output_dir "$OUTPUT_DIR"
    --model_name "$MODEL_NAME"
    --num_samples $NUM_SAMPLES
    --temperature $TEMPERATURE
    --max_new_tokens $MAX_NEW_TOKENS
    --gpu_ids "$GPU_IDS"
)

# Append optional flags.
[ -n "$PRED_BBOX_NORM1000" ] && CMD_ARGS+=($PRED_BBOX_NORM1000)
[ -n "$NO_RESUME" ] && CMD_ARGS+=($NO_RESUME)

# Run generation.
python build_preference_data.py "${CMD_ARGS[@]}"

# Check the result.
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Success: DPO dataset generation finished"
    echo "========================================================================"
    echo "Output files:"
    echo "  - $OUTPUT_DIR/dpo_clean.jsonl"
    echo "  - $OUTPUT_DIR/dpo_pert.jsonl"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "Error: script execution failed"
    echo "========================================================================"
    exit 1
fi
