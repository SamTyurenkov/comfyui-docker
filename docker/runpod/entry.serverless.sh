#!/bin/bash
set -x

export PATH="/workspace/venv_cc12_cuda129/bin:$PATH"
rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py

cp /workspace/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
rm -rf /home/comfyuser/ComfyUI/models/vae_approx && ln -s /workspace/models/vae_approx /home/comfyuser/ComfyUI/models/vae_approx
rm -rf /home/comfyuser/ComfyUI/models/facerestore_models && ln -s /workspace/models/facerestore_models /home/comfyuser/ComfyUI/models/facerestore_models
rm -rf /home/comfyuser/ComfyUI/models/facedetection && ln -s /workspace/models/facedetection /home/comfyuser/ComfyUI/models/facedetection
rm -rf /home/comfyuser/ComfyUI/models/liveportrait && ln -s /workspace/models/liveportrait /home/comfyuser/ComfyUI/models/liveportrait
rm -rf /home/comfyuser/ComfyUI/models/reactor && ln -s /workspace/models/reactor /home/comfyuser/ComfyUI/models/reactor
rm -rf /home/comfyuser/ComfyUI/models/xlabs && ln -s /workspace/models/xlabs /home/comfyuser/ComfyUI/models/xlabs
rm -rf /home/comfyuser/ComfyUI/models/BiRefNet && ln -s /workspace/models/BiRefNet /home/comfyuser/ComfyUI/models/BiRefNet
rm -rf /home/comfyuser/ComfyUI/models/onnx && ln -s /workspace/models/onnx /home/comfyuser/ComfyUI/models/onnx
rm -rf /home/comfyuser/ComfyUI/models/LLM && ln -s /workspace/models/LLM /home/comfyuser/ComfyUI/models/LLM
rm -rf /home/comfyuser/ComfyUI/models/insightface && ln -s /workspace/models/insightface /home/comfyuser/ComfyUI/models/insightface
rm -rf /home/comfyuser/ComfyUI/models/ipadapter && ln -s /workspace/models/ipadapter /home/comfyuser/ComfyUI/models/ipadapter
rm -rf /home/comfyuser/ComfyUI/models/ultralytics && ln -s /workspace/models/ultralytics /home/comfyuser/ComfyUI/models/ultralytics
rm -rf /home/comfyuser/ComfyUI/models/sams && ln -s /workspace/models/sams /home/comfyuser/ComfyUI/models/sams
rm -rf /home/comfyuser/ComfyUI/models/sam2 && ln -s /workspace/models/sam2 /home/comfyuser/ComfyUI/models/sam2
rm -rf /home/comfyuser/ComfyUI/models/text_encoders && ln -s /workspace/models/text_encoders /home/comfyuser/ComfyUI/models/text_encoders
rm -rf /home/comfyuser/ComfyUI/models/clip_vision && ln -s /workspace/models/clip_vision /home/comfyuser/ComfyUI/models/clip_vision
rm -rf /home/comfyuser/ComfyUI/models/stablesr && ln -s /workspace/models/stablesr /home/comfyuser/ComfyUI/models/stablesr
rm -rf /home/comfyuser/ComfyUI/models/style_models && ln -s /workspace/models/style_models /home/comfyuser/ComfyUI/models/style_models

rm -rf /home/comfyuser/ComfyUI/user && ln -s /workspace/user /home/comfyuser/ComfyUI/user

# Start ComfyUI
start_comfyui() {
    # Capture additional arguments from environment variables
    echo "Starting ComfyUI..."
    /workspace/venv_cc12_cuda129/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-sage-attention --disable-xformers --fast fp16_accumulation
}

start_comfyui