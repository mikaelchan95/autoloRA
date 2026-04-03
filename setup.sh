#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo ""
echo "=========================================="
echo "  AutoLoRA Setup"
echo "=========================================="
echo ""

# ── Python ────────────────────────────────────────────────────────────────────

if ! command -v python3 &>/dev/null; then
    fail "python3 not found. Install Python 3.10+."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python $PYTHON_VERSION"

# ── CUDA ──────────────────────────────────────────────────────────────────────

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    info "GPU: $GPU_NAME ($GPU_MEM)"
else
    warn "nvidia-smi not found. GPU training will not work without CUDA."
    warn "Continuing setup anyway (useful for development/testing)."
fi

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
    info "PyTorch CUDA: $TORCH_CUDA"
else
    warn "PyTorch CUDA not available. Install PyTorch with CUDA support:"
    warn "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
fi

# ── Clone Ostris AI Toolkit ───────────────────────────────────────────────────

if [ -d "$ROOT/ai-toolkit" ]; then
    info "ai-toolkit already cloned"
else
    echo ""
    info "Cloning ostris/ai-toolkit..."
    git clone https://github.com/ostris/ai-toolkit.git "$ROOT/ai-toolkit"
    info "ai-toolkit cloned"
fi

# ── Install ai-toolkit deps ──────────────────────────────────────────────────

if [ -f "$ROOT/ai-toolkit/requirements.txt" ]; then
    echo ""
    info "Installing ai-toolkit dependencies..."
    pip install -r "$ROOT/ai-toolkit/requirements.txt"
    info "ai-toolkit deps installed"
fi

# ── Install AutoLoRA deps ────────────────────────────────────────────────────

echo ""
info "Installing AutoLoRA dependencies..."
pip install -r "$ROOT/requirements.txt"
info "AutoLoRA deps installed"

# ── Validate imports ─────────────────────────────────────────────────────────

echo ""
echo "Validating Python imports..."

IMPORTS_OK=true

check_import() {
    if python3 -c "import $1" 2>/dev/null; then
        info "$1"
    else
        warn "$1 — not installed"
        IMPORTS_OK=false
    fi
}

check_import "torch"
check_import "diffusers"
check_import "transformers"
check_import "open_clip"
check_import "insightface"
check_import "yaml"

if python3 -c "from aesthetics_predictor import AestheticsPredictorV2Linear" 2>/dev/null; then
    info "aesthetics_predictor"
else
    warn "aesthetics_predictor — not installed (pip install simple-aesthetics-predictor)"
    IMPORTS_OK=false
fi

if [ "$IMPORTS_OK" = false ]; then
    warn "Some imports failed. The pipeline may not run fully."
fi

# ── Check data directories ───────────────────────────────────────────────────

echo ""

DATASET_COUNT=$(find "$ROOT/dataset" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l | tr -d ' ')
REFERENCE_COUNT=$(find "$ROOT/reference_images" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l | tr -d ' ')

if [ "$DATASET_COUNT" -gt 0 ]; then
    info "dataset/: $DATASET_COUNT images found"
else
    warn "dataset/ is empty. Add training images + .txt captions before running."
    warn "  See dataset/README.md for format details."
fi

if [ "$REFERENCE_COUNT" -gt 0 ]; then
    info "reference_images/: $REFERENCE_COUNT images found"
else
    warn "reference_images/ is empty. Add 8-20 reference images before running."
    warn "  See reference_images/README.md for details."
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  Setup Complete"
echo "=========================================="
echo ""

if [ "$DATASET_COUNT" -eq 0 ] || [ "$REFERENCE_COUNT" -eq 0 ]; then
    echo "Next steps:"
    [ "$DATASET_COUNT" -eq 0 ] && echo "  1. Add training images to dataset/"
    [ "$REFERENCE_COUNT" -eq 0 ] && echo "  2. Add reference images to reference_images/"
    echo "  3. Edit eval_prompts.txt with your trigger word"
    echo "  4. Edit config.yaml model path if not using FLUX.1-dev"
    echo "  5. Run baseline:  python run_experiment.py"
    echo "  6. Start loop:    python run_experiment.py --loop"
else
    echo "Ready to go!"
    echo "  Run baseline:  python run_experiment.py"
    echo "  Start loop:    python run_experiment.py --loop"
fi

echo ""
