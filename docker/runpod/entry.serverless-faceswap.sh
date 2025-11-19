#!/bin/bash
set -x

export PATH="/runpod-volume/venv_cc12_cuda129/bin:$PATH"
rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
mkdir -p /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts
cp /runpod-volume/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
rm -rf /home/comfyuser/ComfyUI/models/vae_approx && ln -s /runpod-volume/models/vae_approx /home/comfyuser/ComfyUI/models/vae_approx
rm -rf /home/comfyuser/ComfyUI/models/facerestore_models && ln -s /runpod-volume/models/facerestore_models /home/comfyuser/ComfyUI/models/facerestore_models
rm -rf /home/comfyuser/ComfyUI/models/facedetection && ln -s /runpod-volume/models/facedetection /home/comfyuser/ComfyUI/models/facedetection
rm -rf /home/comfyuser/ComfyUI/models/liveportrait && ln -s /runpod-volume/models/liveportrait /home/comfyuser/ComfyUI/models/liveportrait
rm -rf /home/comfyuser/ComfyUI/models/reactor && ln -s /runpod-volume/models/reactor /home/comfyuser/ComfyUI/models/reactor
rm -rf /home/comfyuser/ComfyUI/models/xlabs && ln -s /runpod-volume/models/xlabs /home/comfyuser/ComfyUI/models/xlabs
rm -rf /home/comfyuser/ComfyUI/models/BiRefNet && ln -s /runpod-volume/models/BiRefNet /home/comfyuser/ComfyUI/models/BiRefNet
rm -rf /home/comfyuser/ComfyUI/models/onnx && ln -s /runpod-volume/models/onnx /home/comfyuser/ComfyUI/models/onnx
rm -rf /home/comfyuser/ComfyUI/models/LLM && ln -s /runpod-volume/models/LLM /home/comfyuser/ComfyUI/models/LLM
rm -rf /home/comfyuser/ComfyUI/models/insightface && ln -s /runpod-volume/models/insightface /home/comfyuser/ComfyUI/models/insightface
rm -rf /home/comfyuser/ComfyUI/models/ipadapter && ln -s /runpod-volume/models/ipadapter /home/comfyuser/ComfyUI/models/ipadapter
rm -rf /home/comfyuser/ComfyUI/models/ultralytics && ln -s /runpod-volume/models/ultralytics /home/comfyuser/ComfyUI/models/ultralytics
rm -rf /home/comfyuser/ComfyUI/models/sams && ln -s /runpod-volume/models/sams /home/comfyuser/ComfyUI/models/sams
rm -rf /home/comfyuser/ComfyUI/models/sam2 && ln -s /runpod-volume/models/sam2 /home/comfyuser/ComfyUI/models/sam2
rm -rf /home/comfyuser/ComfyUI/models/text_encoders && ln -s /runpod-volume/models/text_encoders /home/comfyuser/ComfyUI/models/text_encoders
rm -rf /home/comfyuser/ComfyUI/models/clip_vision && ln -s /runpod-volume/models/clip_vision /home/comfyuser/ComfyUI/models/clip_vision
rm -rf /home/comfyuser/ComfyUI/models/stablesr && ln -s /runpod-volume/models/stablesr /home/comfyuser/ComfyUI/models/stablesr
rm -rf /home/comfyuser/ComfyUI/models/style_models && ln -s /runpod-volume/models/style_models /home/comfyuser/ComfyUI/models/style_models
rm -rf /home/comfyuser/ComfyUI/models/grounding-dino && ln -s /runpod-volume/models/grounding-dino /home/comfyuser/ComfyUI/models/grounding-dino
rm -rf /home/comfyuser/ComfyUI/user && ln -s /runpod-volume/user /home/comfyuser/ComfyUI/user
rm -rf /home/comfyuser/ComfyUI/input && ln -s /runpod-volume/serverless-input /home/comfyuser/ComfyUI/input
rm -rf /home/comfyuser/ComfyUI/models/hyperswap && ln -s /runpod-volume/models/hyperswap /home/comfyuser/ComfyUI/models/hyperswap
mkdir -p /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts && rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts && ln -s /runpod-volume/models/comfyui_controlnet_aux/ckpts /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts

# Start ComfyUI
start_comfyui() {
    # Set memory optimization environment variables
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export CUDA_LAUNCH_BLOCKING=0
    export CUBLAS_WORKSPACE_CONFIG=:16:8
    # Capture additional arguments from environment variables
    echo "Starting ComfyUI..."
    /runpod-volume/venv_cc12_cuda129/bin/python -u /home/comfyuser/ComfyUI/main.py --max-upload-size 20 --dont-print-server --preview-method taesd --enable-cors-header "*" --disable-xformers --fast fp16_accumulation --disable-auto-launch --disable-metadata --log-stdout
}

start_handler() {
    # Capture additional arguments from environment variables
    echo "Starting Handler..."
    cd /home/comfyuser
    /runpod-volume/venv_cc12_cuda129/bin/python -u handler.py
}

cleanup_old_folders() {
    echo "Starting cleanup of old date folders..."
    
    # Safety check: ensure we're only working in the correct directory
    CLEANUP_DIR="/runpod-volume/serverless-input"
    
    if [ ! -d "$CLEANUP_DIR" ]; then
        echo "Warning: Cleanup directory $CLEANUP_DIR does not exist, skipping cleanup"
        return
    fi
    
    # Calculate date 2 days ago in YYYYMMDD format
    TWO_DAYS_AGO=$(date -d "2 days ago" +%Y%m%d)
    echo "Cleaning up folders older than ${TWO_DAYS_AGO} in $CLEANUP_DIR..."
    
    # Find and remove date folders older than 2 days
    # Extra safety: ensure folder path starts with our cleanup directory
    find "$CLEANUP_DIR" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]" | while read folder; do
        # Double-check the folder is within our target directory
        if [[ "$folder" == "$CLEANUP_DIR"/* ]] && [ -d "$folder" ]; then
            folder_date=$(basename "$folder")
            # Validate the folder name is exactly 8 digits
            if [[ "$folder_date" =~ ^[0-9]{8}$ ]] && [ "$folder_date" -lt "$TWO_DAYS_AGO" ]; then
                echo "Removing old folder: $folder"
                rm -rf "$CLEANUP_DIR/$folder_date"
            fi
        fi
    done
    echo "Cleanup completed."
}

# Start all services in parallel
start_comfyui &
cleanup_old_folders &
start_handler