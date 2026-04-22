#!/bin/bash




# ============================================
# MuKV - RVS-Ego

# Main method: multi-grained KV cache + dual-signal compression + rerank retrieval
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"


export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "MuKV"
echo "=========================================="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Time: $(date)"
echo "Dataset: RVS-Ego"
echo "Method: MuKV main experiment"
echo "=========================================="
echo ""

export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

# ============================================
# Configurable Parameters
# ============================================

# Model and Dataset
MODEL_PATH="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
DATASET="rvs_ego"

# Auto-detect model name from path
if [[ "$MODEL_PATH" == *"0.5b"* ]]; then
    MODEL_NAME="llava_ov_0.5b"
elif [[ "$MODEL_PATH" == *"7b"* ]]; then
    MODEL_NAME="llava_ov_7b"
else
    MODEL_NAME="llava_ov_unknown"
fi

# Multi-Granularity Parameters
GRANULARITIES="49 196 784"
TOPKS="20 32 12"  # Total=64

# Token Compression Parameters
ENABLE_COMPRESSION="true"
KEEP_RATIO_49=0.1
KEEP_RATIO_196=0.1
KEEP_RATIO_784=0.8

IMPORTANCE_METHOD="attention_fft_weighted"
DEDUP_ENABLED="false"
DEDUP_THRESHOLD=0.90

# FFT Parameters
FFT_ENABLED="true"
FFT_METHOD="fft"  # diff | fft | spectral_entropy
ATTENTION_WEIGHT_49=0.5   # alpha for granularity 49
ATTENTION_WEIGHT_196=0.7  # alpha for granularity 196
ATTENTION_WEIGHT_784=0.8  # alpha for granularity 784
# FFT weight beta = 1 - alpha

# Attention Layer Strategy
ATTENTION_LAYER_STRATEGY="last"  # last | second_last | last_5 | middle

# Unified Layer Strategy (FFT和Query也使用与Attention相同的层)
USE_UNIFIED_LAYER="false"  # true | false

# Rerank Parameters
ENABLE_RERANK="true"
RERANK_ALPHA=0.3   # Medium-grain: original relevance vs scene consistency
RERANK_BETA=0.3    # Fine-grain: original relevance vs scene consistency
RERANK_TOP_N=5     # Top-N coarse blocks for context definition
RERANK_REVERSE="false"  # 🔄 Reverse mode: false=前20/32/12 (正常), true=后20/32/12 (反向测试)

# Video format (mp4 | npy)
VIDEO_FORMAT="mp4"  # npy format loads faster

# ============================================
# Generate Save Directory
# ============================================

# Method name
METHOD_NAME="mukv"

# Generate weight identifier (attention weight: alpha)
WEIGHT1=$(echo $ATTENTION_WEIGHT_49 | awk '{printf "%02d", $1*100}')
WEIGHT2=$(echo $ATTENTION_WEIGHT_196 | awk '{printf "%02d", $1*100}')
WEIGHT3=$(echo $ATTENTION_WEIGHT_784 | awk '{printf "%02d", $1*100}')
WEIGHT_STR="a${WEIGHT1}_${WEIGHT2}_${WEIGHT3}"

# Generate keep ratio identifier
RATIO1=$(echo $KEEP_RATIO_49 | awk '{printf "%02d", $1*100}')
RATIO2=$(echo $KEEP_RATIO_196 | awk '{printf "%02d", $1*100}')
RATIO3=$(echo $KEEP_RATIO_784 | awk '{printf "%02d", $1*100}')
RATIO_STR="r${RATIO1}_${RATIO2}_${RATIO3}"

# Generate Rerank identifier
RERANK_ALPHA_INT=$(echo $RERANK_ALPHA | awk '{printf "%02d", $1*100}')
RERANK_BETA_INT=$(echo $RERANK_BETA | awk '{printf "%02d", $1*100}')
RERANK_STR="rr_a${RERANK_ALPHA_INT}_b${RERANK_BETA_INT}_n${RERANK_TOP_N}"

# Add reverse mode suffix
if [ "$RERANK_REVERSE" = "true" ]; then
    RERANK_STR="${RERANK_STR}_REV"  # 添加_REV标识反向模式
fi

# Timestamp
TIMESTAMP=$(date +"%m%d_%H%M")

# FFT method suffix
if [ "$FFT_METHOD" = "diff" ]; then
    FFT_SUFFIX="diff"
elif [ "$FFT_METHOD" = "fft" ]; then
    FFT_SUFFIX="fft"
elif [ "$FFT_METHOD" = "spectral_entropy" ]; then
    FFT_SUFFIX="se"
else
    FFT_SUFFIX="unknown"
fi

# Attention layer strategy suffix
if [ "$ATTENTION_LAYER_STRATEGY" = "last" ]; then
    LAYER_SUFFIX="L1"
elif [ "$ATTENTION_LAYER_STRATEGY" = "second_last" ]; then
    LAYER_SUFFIX="L2"
elif [ "$ATTENTION_LAYER_STRATEGY" = "last_5" ]; then
    LAYER_SUFFIX="L5"
elif [ "$ATTENTION_LAYER_STRATEGY" = "middle" ]; then
    LAYER_SUFFIX="Lm"
else
    LAYER_SUFFIX="L1"
fi

# Unified layer suffix
if [ "$USE_UNIFIED_LAYER" = "true" ]; then
    LAYER_SUFFIX="${LAYER_SUFFIX}_unified"
fi

# Final directory
SAVE_DIR="results/${METHOD_NAME}/${DATASET}/${MODEL_NAME}/${TIMESTAMP}_${WEIGHT_STR}_${RATIO_STR}_${RERANK_STR}_${FFT_SUFFIX}_${LAYER_SUFFIX}"
mkdir -p $SAVE_DIR

echo "Rerank Configuration:"
echo "  Enable: $ENABLE_RERANK"
echo "  Alpha (medium-grain): $RERANK_ALPHA"
echo "  Beta (fine-grain): $RERANK_BETA"
echo "  Top-N for context: $RERANK_TOP_N"
echo "  🔄 Reverse Mode: $RERANK_REVERSE"
echo ""
echo "FFT Configuration:"
echo "  FFT Enabled: $FFT_ENABLED"
echo "  FFT Method: $FFT_METHOD"
echo "  Attention Layer Strategy: $ATTENTION_LAYER_STRATEGY"
echo "  Use Unified Layer: $USE_UNIFIED_LAYER"
echo "  Attention Weights (alpha): 49->${ATTENTION_WEIGHT_49}, 196->${ATTENTION_WEIGHT_196}, 784->${ATTENTION_WEIGHT_784}"
echo "  FFT Weights (beta=1-alpha): 49->$((100-$(echo $ATTENTION_WEIGHT_49 | awk '{printf "%d", $1*100}')))%, 196->$((100-$(echo $ATTENTION_WEIGHT_196 | awk '{printf "%d", $1*100}')))%, 784->$((100-$(echo $ATTENTION_WEIGHT_784 | awk '{printf "%d", $1*100}')))%"
echo "  Keep Ratios: 49->${KEEP_RATIO_49}, 196->${KEEP_RATIO_196}, 784->${KEEP_RATIO_784}"
echo "  Dedup: $DEDUP_ENABLED"
echo "  Video Format: $VIDEO_FORMAT"
echo "  Save Directory: $SAVE_DIR"
echo ""

# Run inference
python scripts/run_mukv_rvs_ego.py \
  --model_path ${MODEL_PATH} \
  --anno_path "data/rvs/ego/ego4d_oe.json" \
  --sample_fps 0.5 \
  --n_local 15000 \
  --retrieve_size 64 \
  --retrieve_chunk_size 1 \
  --save_dir $SAVE_DIR \
  --num_chunks 1 \
  --chunk_idx 0 \
  --granularities ${GRANULARITIES} \
  --granularity_topks ${TOPKS} \
  --enable_compression ${ENABLE_COMPRESSION} \
  --keep_ratio_49 ${KEEP_RATIO_49} \
  --keep_ratio_196 ${KEEP_RATIO_196} \
  --keep_ratio_784 ${KEEP_RATIO_784} \
  --importance_method ${IMPORTANCE_METHOD} \
  --dedup_threshold ${DEDUP_THRESHOLD} \
  --dedup_enabled ${DEDUP_ENABLED} \
  --fft_enabled ${FFT_ENABLED} \
  --fft_method ${FFT_METHOD} \
  --attention_weight_49 ${ATTENTION_WEIGHT_49} \
  --attention_weight_196 ${ATTENTION_WEIGHT_196} \
  --attention_weight_784 ${ATTENTION_WEIGHT_784} \
  --attention_layer_strategy ${ATTENTION_LAYER_STRATEGY} \
  --use_unified_layer ${USE_UNIFIED_LAYER} \
  --video_format ${VIDEO_FORMAT} \
  --enable_rerank ${ENABLE_RERANK} \
  --rerank_alpha ${RERANK_ALPHA} \
  --rerank_beta ${RERANK_BETA} \
  --rerank_top_n ${RERANK_TOP_N} \
  --rerank_reverse ${RERANK_REVERSE}

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Inference completed! Starting GPT evaluation..."
echo "=========================================="

if [ $EXIT_CODE -ne 0 ]; then
    echo "Inference failed, exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

# GPT evaluation
echo "Starting GPT-3.5 evaluation..."
python video_qa/eval/eval_open_ended.py \
  --pred_path ${SAVE_DIR}/results.csv \
  --output_dir ${SAVE_DIR}/gpt_eval \
  --output_json ${SAVE_DIR}/results.json \
  --num_tasks 8

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Inference: ${SAVE_DIR}/results.csv"
echo "  Scores: ${SAVE_DIR}/results.json"
echo "  Memory: ${SAVE_DIR}/memory_stats.csv"
echo "  Log: ${SAVE_DIR}/inference.log"
echo "  Save Directory: ${SAVE_DIR}"
echo "=========================================="
