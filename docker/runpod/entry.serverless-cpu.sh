#!/bin/bash
set -x

export PATH="/runpod-volume/venv_cc12_cuda129/bin:$PATH"
export OMP_NUM_THREADS=4
export ONNXRT_NUM_THREADS=4

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

mkdir -p /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger/models && rm -rf /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger/models && ln -s /runpod-volume/models/ComfyUI-WD14-Tagger/models /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger/models
mkdir -p /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts && rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts && ln -s /runpod-volume/models/comfyui_controlnet_aux/ckpts /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts


# Start ComfyUI
start_comfyui() {
    # Capture additional arguments from environment variables
    echo "Starting ComfyUI..."
    /runpod-volume/venv_cc12_cuda129/bin/python -u /home/comfyuser/ComfyUI/main.py --max-upload-size 20 --dont-print-server --enable-cors-header "*" --cpu --disable-auto-launch --disable-metadata --log-stdout
}

start_handler() {
    # Capture additional arguments from environment variables
    echo "Starting Handler..."
    /runpod-volume/venv_cc12_cuda129/bin/python -u /home/comfyuser/handler.py
}

start_comfyui &
start_handler