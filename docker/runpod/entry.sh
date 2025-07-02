#!/bin/bash
set -x

# TODO: FINISH GOOGLE DRIVE SERVICE ACCOUNT LOGIC

# client_id = YOUR_GOOGLE_DRIVE_CLIENT_ID
# client_secret = YOUR_GOOGLE_DRIVE_CLIENT_SECRET

mkdir -p /root/.config/rclone
cat >/root/.config/rclone/rclone.conf <<EOL
[runpod]
type = s3
provider = Runpod
access_key_id = ${RUNPOD_USERID}
secret_access_key = ${RUNPOD_TOKEN}
endpoint = https://s3api-eu-ro-1.runpod.io
acl = private
EOL

mkdir -p /root/.aws/
cat >/root/.aws/credentials <<EOL
[default]
aws_access_key_id = ${RUNPOD_USERID}
aws_secret_access_key = ${RUNPOD_TOKEN}
EOL

if [ -n "$ROAMING_WAN" ] && [ "$ROAMING_WAN" = "1" ]; then
  wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update && apt-get -y install cuda-toolkit-12-9
  export CUDA_HOME="/usr/local/cuda-12.9" && export PATH="$CUDA_HOME/bin:$PATH" && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64"

  pip install --upgrade pip
  pip install torch==2.7.0 protobuf==4.25.3 numpy==1.26.4 torchvision torchaudio torchsde --extra-index-url https://download.pytorch.org/whl/cu128
  pip install diffusers aiohttp aiodns Brotli flet==0.27.6 matplotlib-inline albumentations==2.0.8 transparent-background
  pip install simsimd --prefer-binary
  pip install setuptools wheel build triton spandrel kornia av jedi==0.16 onnxruntime tf-keras==2.19.0
  pip install -r /home/comfyuser/ComfyUI/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-subpack/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-art-venture/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-easy-use/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/efficiency-nodes-comfyui/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SAM2/requirements.txt
  pip install -r /home/comfyuser/ComfyUI/custom_nodes/teacache/requirements.txt
  pip install -e /home/comfyuser/sageattention/. --use-pep517 --verbose
  python /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/install.py
  pip install --force-reinstall --no-deps numpy==1.26.4
  pip uninstall -y tensorflow
elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
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

cp /home/comfyuser/docker/nginx/nginx.conf /etc/nginx/nginx.conf
cp /home/comfyuser/docker/nginx/site-conf/default.conf /etc/nginx/conf.d/default.conf


rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py

if [ -n "$ROAMING_WAN" ] && [ "$ROAMING_WAN" = "1" ]; then

  mkdir -p /home/comfyuser/ComfyUI/models/vae
  mkdir -p /home/comfyuser/ComfyUI/models/vae_approx
  mkdir -p /home/comfyuser/ComfyUI/models/loras/wan
  mkdir -p /home/comfyuser/ComfyUI/models/clip_vision
  mkdir -p /home/comfyuser/ComfyUI/models/text_encoders
  mkdir -p /home/comfyuser/ComfyUI/models/diffusion_models

  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/

  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/vae/wan_2.1_vae_bf16.safetensors /home/comfyuser/ComfyUI/models/vae/
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/vae/wan_2.1_vae.safetensors /home/comfyuser/ComfyUI/models/vae/
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/vae_approx /home/comfyuser/ComfyUI/models/vae_approx
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan /home/comfyuser/ComfyUI/models/loras/wan
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/clip_vision /home/comfyuser/ComfyUI/models/clip_vision
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/text_encoders/umt5-xxl-enc-bf16.safetensors /home/comfyuser/ComfyUI/models/text_encoders/
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/text_encoders/clip_l.safetensors /home/comfyuser/ComfyUI/models/text_encoders/

  if [ -n "$LOAD_FUN_INP_MODEL" ] && [ "$LOAD_FUN_INP_MODEL" = "1" ]; then
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/Wan2.1-Fun-14B-InP.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/Wan2.1-Fun-14B-InP.safetensors
  fi

  if [ -n "$LOAD_DMD_VACE_MODEL" ] && [ "$LOAD_DMD_VACE_MODEL" = "1" ]; then
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/Wan2.1-T2V-1.3B-Self-Forcing-DMD-FP16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/Wan2.1-T2V-1.3B-Self-Forcing-DMD-FP16.safetensors
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/Wan2.1-T2V-1.3B-Self-Forcing-DMD-VACE-FP16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/Wan2.1-T2V-1.3B-Self-Forcing-DMD-VACE-FP16.safetensors
  fi

  if [ -n "$LOAD_I2V_MODEL" ] && [ "$LOAD_I2V_MODEL" = "1" ]; then
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors
  fi

  if [ -n "$LOAD_FLF2V_MODEL" ] && [ "$LOAD_FLF2V_MODEL" = "1" ]; then
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/wan2.1_flf2v_720p_14B_fp16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/wan2.1_flf2v_720p_14B_fp16.safetensors
  fi

  if [ -n "$LOAD_VACE_MODEL" ] && [ "$LOAD_VACE_MODEL" = "1" ]; then
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/wan2.1_vace_14B_fp16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp16.safetensors
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/Wan2_1-VACE_module_14B_bf16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/Wan2_1-VACE_module_14B_bf16.safetensors
  fi

  if [ -n "$LOAD_FUSIONX_MODEL" ] && [ "$LOAD_FUSIONX_MODEL" = "1" ]; then
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/Wan14BT2VFusioniX_Phantom_fp16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/Wan14BT2VFusioniX_Phantom_fp16.safetensors
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/diffusion_models/Wan14Bi2vFusioniX_fp16.safetensors /home/comfyuser/ComfyUI/models/diffusion_models/Wan14Bi2vFusioniX_fp16.safetensors
  fi

else
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

  rm -rf /home/comfyuser/ComfyUI/user && ln -s /workspace/user /home/comfyuser/ComfyUI/user
fi

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
    if [ -n "$ROAMING_WAN" ] && [ "$ROAMING_WAN" = "1" ]; then
      python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
      /workspace/venv_cc12_cuda129/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "CPU" ]; then
      /workspace/venv/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --preview-method taesd --enable-cors-header "*" --cpu
    else
      #default CC 8.0
      /workspace/venv/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
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
