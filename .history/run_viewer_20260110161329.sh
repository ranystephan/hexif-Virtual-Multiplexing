#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# HEXIF Core Viewer Launch Script
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage examples:
#
#   1. Basic usage (just data, no model):
#      ./run_viewer.sh --pairs_dir /path/to/core_patches_npy
#
#   2. With model for predictions:
#      ./run_viewer.sh --pairs_dir /path/to/core_patches_npy \
#                      --checkpoint runs/nov5/focal_l1_plateau/best_model.pth
#
#   3. Custom port (useful for code-server):
#      ./run_viewer.sh --port 8502
#
# ─────────────────────────────────────────────────────────────────────────────

# Default values
PAIRS_DIR="core_patches_npy"
CHECKPOINT=""
PORT="8501"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pairs_dir)
            PAIRS_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔬 HEXIF Core Viewer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Data directory: $PAIRS_DIR"
echo "Checkpoint: ${CHECKPOINT:-'None (view-only mode)'}"
echo "Port: $PORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting Streamlit server..."
echo "Access the viewer at: http://localhost:$PORT"
echo ""

# Run streamlit
cd "$SCRIPT_DIR"
streamlit run viewer.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
