#!/usr/bin/env bash
# Frontier-scale replication of the taxonomy protocol on a 70B model.
# Runs on a vast.ai GPU instance with sufficient VRAM (~140GB for 70B in fp16,
# or use bitsandbytes 4-bit quantization to fit in 48GB).
#
# Pre-flight on your laptop:
#   vastai create instance --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \\
#       --disk 200 --gpu-name "RTX A6000" --gpu-count 2  # for 70B fp16
#   # or --gpu-name "H100" --gpu-count 1 for fp16 on 80GB
#   # or --gpu-name "RTX 4090" --gpu-count 1 --bnb-4bit for 4-bit quantization
#
# Then on the instance:
#   git clone <your repo> ; cd <project>
#   git checkout claude/silly-goldberg-7eeb73
#   bash run_70b_vastai.sh
#
# Output: taxonomy_reports_llama70b/ (or taxonomy_reports_qwen72b/) -- a
# new (model, task) bundle the LOO CV script can pool with the existing 48.
# Expected runtime: ~12 hours for the full 12-task corpus at N=200 pairs
# on 1xH100, longer on quantized GPUs.

set -euo pipefail

# Edit these for your run
MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
LAYER="${LAYER:-20}"     # ~25% depth on 80-layer Llama-70B
OUT_DIR="${OUT_DIR:-taxonomy_reports_llama70b}"
N_PAIRS="${N_PAIRS:-200}"
N_EVAL="${N_EVAL:-100}"
N_RANDOM="${N_RANDOM:-5}"
SEED="${SEED:-2026}"

# Quantization off by default. For 24/48GB GPUs, set BNB_4BIT=1.
BNB_4BIT="${BNB_4BIT:-0}"

echo "=== Frontier-scale taxonomy run ==="
echo "Model:     $MODEL"
echo "Layer:     $LAYER"
echo "OutDir:    $OUT_DIR"
echo "N pairs:   $N_PAIRS"
echo "N eval:    $N_EVAL"
echo "Quantized: $BNB_4BIT"
echo ""

# Setup env (vast.ai instances ship with conda + pip)
pip install -q --upgrade transformers accelerate datasets tqdm
if [ "$BNB_4BIT" = "1" ]; then
    pip install -q bitsandbytes
fi

mkdir -p "$OUT_DIR"

# Patch the runner for bnb-4bit if requested (only if BNB_4BIT=1).
# This is a temporary in-place edit local to the vast.ai instance; the
# committed runner.py is unchanged.
if [ "$BNB_4BIT" = "1" ]; then
    python - <<'PY'
import re, pathlib
p = pathlib.Path("steering_taxonomy/runner.py")
src = p.read_text()
patched = src.replace(
    'self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)',
    'from transformers import BitsAndBytesConfig\n'
    '        bnb_config = BitsAndBytesConfig(load_in_4bit=True,\n'
    '            bnb_4bit_compute_dtype=torch.float16,\n'
    '            bnb_4bit_quant_type="nf4")\n'
    '        self.model = AutoModelForCausalLM.from_pretrained(model_id,\n'
    '            quantization_config=bnb_config, device_map="auto")'
)
p.write_text(patched)
print("Patched runner for 4-bit quantization")
PY
fi

# Run the full 12-task corpus
python -m steering_taxonomy.run \
    --model "$MODEL" \
    --layer "$LAYER" \
    --n-pairs "$N_PAIRS" \
    --n-eval "$N_EVAL" \
    --n-random "$N_RANDOM" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "${OUT_DIR}.log"

echo ""
echo "=== Done. Reports in $OUT_DIR/ ==="
echo ""
echo "Sync back to laptop with:"
echo "  rsync -av --progress vast-instance:$OUT_DIR/ ./$OUT_DIR/"
echo "  rsync -av --progress vast-instance:${OUT_DIR}.log ./"
echo ""
echo "Then on laptop, re-run LOO CV to update the pooled accuracy:"
echo "  # Edit loo_cv.py to add the new model to MODELS dict:"
echo "  #   \"llama-70b\": \"$OUT_DIR\","
echo "  python3 loo_cv.py --pooled --asymmetry --out-json loo_cv_5model.json"
