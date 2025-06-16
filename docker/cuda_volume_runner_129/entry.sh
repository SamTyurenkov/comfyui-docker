#!/bin/bash
set -x
export PATH="/workspace/venv/bin:$PATH"
which python
which pip
echo $CUDA_HOME

cp /home/comfyuser/docker/nginx/nginx.conf /etc/nginx/nginx.conf
cp /home/comfyuser/docker/nginx/site-conf/default.conf /etc/nginx/conf.d/default.conf

rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
cp /workspace/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py

# reactor breaks loading often
mv /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node /tmp/comfyui-reactor-node

rm -rf /home/comfyuser/ComfyUI/models/vae_approx
rm -rf /home/comfyuser/ComfyUI/models/text_encoders
rm -rf /home/comfyuser/ComfyUI/models/clip_vision
# rclone copy --log-level=INFO r2:europe-colab/extra_model_paths.yaml /home/comfyuser/ComfyUI/
ln -s /workspace/models/vae_approx /home/comfyuser/ComfyUI/models/vae_approx
ln -s /workspace/models/facerestore_models /home/comfyuser/ComfyUI/models/facerestore_models
ln -s /workspace/models/facedetection /home/comfyuser/ComfyUI/models/facedetection
ln -s /workspace/models/liveportrait /home/comfyuser/ComfyUI/models/liveportrait
ln -s /workspace/models/reactor /home/comfyuser/ComfyUI/models/reactor
ln -s /workspace/models/xlabs /home/comfyuser/ComfyUI/models/xlabs
ln -s /workspace/models/BiRefNet /home/comfyuser/ComfyUI/models/BiRefNet
ln -s /workspace/models/onnx /home/comfyuser/ComfyUI/models/onnx
ln -s /workspace/models/LLM /home/comfyuser/ComfyUI/models/LLM
ln -s /workspace/models/insightface /home/comfyuser/ComfyUI/models/insightface
ln -s /workspace/models/ipadapter /home/comfyuser/ComfyUI/models/ipadapter
ln -s /workspace/models/ultralytics /home/comfyuser/ComfyUI/models/ultralytics
ln -s /workspace/models/text_encoders /home/comfyuser/ComfyUI/models/text_encoders
ln -s /workspace/models/clip_vision /home/comfyuser/ComfyUI/models/clip_vision

mkdir -p /workspace/user
rm -rf /home/comfyuser/ComfyUI/user
ln -s /workspace/user /home/comfyuser/ComfyUI/user

# Start cloudflared in the background
start_cloudflared() {
    echo "Starting cloudflared..."
    echo "Using tunnel token and name..."
    cloudflared tunnel \
        --loglevel info \
        --logfile /var/log/cloudflared.log \
        run --token "$TUNNEL_TOKEN" "$TUNNEL_NAME"
}

# Start ComfyUI
start_comfyui() {
    echo "Starting ComfyUI..."
    if [ -n "$COMMAND" ]; then
      exec sh -c "$COMMAND"
    else
      /workspace/venv/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 100 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers
    fi
}

start_nginx() {
  echo "Starting Nginx..."
  nginx -g "daemon off;"
}

start_cloudflared &
start_comfyui &
start_nginx