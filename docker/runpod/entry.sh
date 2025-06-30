#!/bin/bash
set -x
# mkdir -p /root/.config/rclone

# cat << EOF > /root/.config/rclone/rclone.conf
# [r2]
# type = s3
# provider = Cloudflare
# access_key_id = $R2_ID
# secret_access_key = $R2_KEY
# endpoint = https://c45a8b257df7bbf1fb0c4fb396042547.r2.cloudflarestorage.com
# acl = private

# [google-drive]
# type = drive
# client_id =
# client_secret =
# scope = drive
# token = $GOOGLE_DRIVE_TOKEN_JSON
# EOF
# TODO: FINISH GOOGLE DRIVE SERVICE ACCOUNT LOGIC

# client_id = YOUR_GOOGLE_DRIVE_CLIENT_ID
# client_secret = YOUR_GOOGLE_DRIVE_CLIENT_SECRET
# pip install --upgrade pip
# pip install torch==2.7.0 protobuf==4.25.3 numpy==1.26.4 torchvision torchaudio torchsde --extra-index-url https://download.pytorch.org/whl/cu128
# pip install diffusers aiohttp aiodns Brotli flet==0.27.6 matplotlib-inline albumentations==2.0.8 transparent-background xformers insightface
# pip install simsimd --prefer-binary
# pip install setuptools wheel build triton spandrel kornia av jedi==0.16 onnxruntime tf-keras==2.19.0

if [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
  export PATH="/workspace/venv_cc12_cuda129/bin:$PATH"
elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "8.9" ]; then
  export PATH="/workspace/venv_cc8_9_cuda129/bin:$PATH"
elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "CPU" ]; then
    export PATH="/workspace/venv/bin:$PATH"
else
  #default CC 8.0
  export PATH="/workspace/venv/bin:$PATH"
fi

which python
which pip

# pip install -r /home/comfyuser/ComfyUI/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-subpack/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-advancedliveportrait/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-LBMWrapper/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-art-venture/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui_chatgpt/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-easy-use/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyI2I/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/efficiency-nodes-comfyui/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfy_mtb/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/pfaeff-comfyui/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SUPIR/requirements.txt
# pip install -r /home/comfyuser/ComfyUI/custom_nodes/teacache/requirements.txt
# # pip install -e /home/comfyuser/sageattention/. --use-pep517 --verbose
# python /home/comfyuser/ComfyUI/custom_nodes/x-flux-comfyui/setup.py
# python /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/install.py
# pip install --force-reinstall --no-deps numpy==1.26.4

# pip install torch==2.7.0 numpy==1.26.4 torchvision torchaudio torchsde --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu128
# python /workspace/sageattention/setup.py install
# python /home/comfyuser/ComfyUI/custom_nodes/x-flux-comfyui/setup.py
# python /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/install.py

cp /home/comfyuser/docker/nginx/nginx.conf /etc/nginx/nginx.conf
cp /home/comfyuser/docker/nginx/site-conf/default.conf /etc/nginx/conf.d/default.conf

# mkdir -p /mnt/r2_cache
# mkdir -p /home/comfyuser/r2_bucket
# rclone mount --daemon --log-level=DEBUG --cache-dir=/mnt/r2_cache --allow-other --vfs-cache-max-age=12h --dir-cache-time=12h --poll-interval=10s --vfs-cache-mode=full --vfs-read-chunk-size=128M --buffer-size=64M --use-server-modtime r2:europe-colab/models /home/comfyuser/r2_bucket 

rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
cp /workspace/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
# rclone copy --log-level=INFO r2:europe-colab/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/

# reactor breaks loading often
# mv /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node /tmp/comfyui-reactor-node

# rclone copy --log-level=INFO r2:europe-colab/extra_model_paths.yaml /home/comfyuser/ComfyUI/
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

rm -rf /home/comfyuser/ComfyUI/user && ln -s /workspace/user /home/comfyuser/ComfyUI/user

# Start cloudflared in the background
start_cloudflared() {
    if [ -n "$TUNNEL_TOKEN" ] && [ "$TUNNEL_TOKEN" != "SET_YOUR_TUNNEL_TOKEN" ]; then
      echo "Starting cloudflared..."
      echo "Using tunnel token and name..."
      cloudflared tunnel \
          --loglevel info \
          --logfile /var/log/cloudflared.log \
          run --token "$TUNNEL_TOKEN" "$TUNNEL_NAME"
    else
      echo "Skipping private cloudflared, tunnel data not specified"
      cloudflared tunnel --url https://127.0.0.1 --loglevel info
    fi
}

# Start ComfyUI
start_comfyui() {
    echo "Starting ComfyUI..."
    if [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
      /workspace/venv_cc12_cuda129/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 100 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "CPU" ]; then
      /workspace/venv/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 100 --dont-print-server --preview-method taesd --enable-cors-header "*" --cpu
    else
      #default CC 8.0
      /workspace/venv/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 100 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    fi
}

start_jupyterlab() {
    echo "Starting Jupyter Lab..."
    jupyter lab --allow-root --no-browser --port=8888 --ip=* --FileContentsManager.delete_to_trash=False --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=*
}

start_nginx() {
  echo "Starting Nginx..."
  nginx -g "daemon off;"
}

start_cloudflared &
start_comfyui &
start_nginx &
start_jupyterlab


# wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb
# apt-get update && apt-get -y install cuda-toolkit-12-9
# export CUDA_HOME="/usr/local/cuda-12.9" && export PATH="$CUDA_HOME/bin:$PATH" && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64"
