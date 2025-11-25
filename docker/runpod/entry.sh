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
  dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
  apt-get update && apt-get -y install cuda-toolkit
  export CUDA_HOME="/usr/local/cuda" && export PATH="$CUDA_HOME/bin:$PATH" && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64"
  git clone https://github.com/thu-ml/SageAttention.git /home/comfyuser/sageattention
  # MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="8.9;12.0" /workspace/venv_cc12_cuda129/bin/python -m pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
  # MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="8.9;12.0" /workspace/venv_cc12_cuda130/bin/python -m pip install flash_attn==2.8.2 --no-build-isolation --verbose
  # MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="8.9;12.0" venv_cc12_cuda130/bin/python -m pip install sageattention==2.2.0 --no-build-isolation --verbose
  # sam2 compile from source
  # torch torchvision torchsde xformers from index url
  pip install --upgrade pip
  pip install torch==2.8.0 protobuf==4.25.3 numpy==1.26.4 torchvision torchaudio torchsde --extra-index-url https://download.pytorch.org/whl/cu128
  pip install diffusers aiohttp aiodns Brotli flet==0.27.6 matplotlib-inline albumentations==2.0.8 transparent-background
  pip install simsimd --prefer-binary
  pip install setuptools polygraphy wheel build triton spandrel kornia av jedi==0.16 onnxruntime tf-keras==2.19.0
  pip install tensorrt --no-build-isolation
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
  pip install -e /home/comfyuser/sageattention/. --use-pep517 --verbose --no-build-isolation
  python /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/install.py
  pip install --force-reinstall --no-deps protobuf==4.25.3 numpy==1.26.4
  pip uninstall -y tensorflow
elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
  export PATH="/workspace/venv_cc12_cuda129/bin:$PATH"
elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "8.9" ]; then
  export PATH="/workspace/venv_cc8_9_cuda129/bin:$PATH"
elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "CPU" ]; then
    export PATH="/workspace/venv_cc12_cuda129/bin:$PATH"
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

  # # ### FLUX TAE DECODER/ENCODER FOR PREVIEWS AND VAE decode/encode
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taef1_decoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taef1_decoder.pth"
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taef1_encoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taef1_encoder.pth"
  # # ### SDXL TAE DECODER/ENCODER FOR PREVIEWS AND VAE decode/encode
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taesdxl_decoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taesdxl_decoder.pth"
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taesdxl_encoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taesdxl_encoder.pth"
  # # ### SD3 TAE DECODER/ENCODER FOR PREVIEWS AND VAE decode/encode
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taesd3_decoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taesd3_decoder.pth"
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taesd3_encoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taesd3_encoder.pth"
  # # ### SD TAE DECODER/ENCODER FOR PREVIEWS AND VAE decode/encode
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taesd_decoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_decoder.pth"
  # wget -O /home/comfyuser/ComfyUI/models/vae_approx/taesd_encoder.pth "https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_encoder.pth"
  
  wget -O /home/comfyuser/ComfyUI/models/vae_approx/taew2_1.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/taew2_1.safetensors?download=true"
  wget -O /home/comfyuser/ComfyUI/models/clip_vision/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp16.safetensors?download=true"
  wget -O /home/comfyuser/ComfyUI/models/clip_vision/clip_vision_h.safetensors "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true"

  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/vae/wan_2.1_vae_bf16.safetensors /home/comfyuser/ComfyUI/models/vae/
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/vae/wan_2.1_vae.safetensors /home/comfyuser/ComfyUI/models/vae/
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

  # aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/ /home/comfyuser/ComfyUI/models/loras/wan --recursive --no-paginate
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors /home/comfyuser/ComfyUI/models/loras/wan/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/pov-missionary-t2v_v1_1_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/pov-missionary-t2v_v1_1_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/detailz_t2v_14B_v1.safetensors /home/comfyuser/ComfyUI/models/loras/wan/detailz_t2v_14B_v1.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/nsfwfemale-genitals-helper-for-wan-t2vi2v.safetensors /home/comfyuser/ComfyUI/models/loras/wan/nsfwfemale-genitals-helper-for-wan-t2vi2v.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/pov-handjob_t2v_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/pov-handjob_t2v_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/pov-missionary-i2v_v1_1_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/pov-missionary-i2v_v1_1_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/dick_slap_T2V_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/dick_slap_T2V_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/pov-blowjob-t2v-and-i2v_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/pov-blowjob-t2v-and-i2v_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/wan_cumshot_t2v_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/wan_cumshot_t2v_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/wan_cumshot_i2v_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/wan_cumshot_i2v_14B.safetensors
  aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/ s3://kns8p9opbh/models/loras/wan/anal_from_behind_t2v_14B.safetensors /home/comfyuser/ComfyUI/models/loras/wan/anal_from_behind_t2v_14B.safetensors
  
# 360-i2v_480p-lora.safetensors                                                     penis-lora-front-view-blowjob-cumshot-taz-wan-21-14b-13b-t2v-and-i2v.sha256  pov-titty-fuck-ti2v-and-i2v_720p_v1.safetensors
# 360-i2v_480p-lora.sha256                                                          pov-blowjob-t2v-and-i2v_14B.safetensors                                      rev3rse-c0wg1rl-wan-21-i2v_720p-lora_v1.safetensors
# Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors                                  pov-blowjob-t2v-and-i2v_14B.sha256                                           rev3rse-c0wg1rl-wan-21-i2v_720p-lora_v1.sha256
# ahegao_face_i2v_720p_t2v_v1.safetensors                                           pov-handjob_i2v720p_v1.safetensors                                           reverse_cowgirl_wan-21_t2v_14B.safetensors
# anal_from_behind_t2v_14B.safetensors                                              pov-handjob_t2v_14B.safetensors                                              singularunity-twerk-wan21_720p-i2v_v2.safetensors
# detailz_t2v_14B_v1.safetensors                                                    pov-handjob_t2v_14B.sha256                                                   wan_cumshot_i2v_14B.safetensors
# detailz_t2v_14B_v1.sha256                                                         pov-missionary-i2v_v1_1_14B.safetensors                                      wan_cumshot_i2v_14B.sha256
# dick_slap_T2V_14B.safetensors                                                     pov-missionary-i2v_v1_1_14B.sha256                                           wan_cumshot_t2v_14B.safetensors
# nsfwfemale-genitals-helper-for-wan-t2vi2v.safetensors                             pov-missionary-t2v_v1_1_14B.safetensors                                      wan_cumshot_t2v_14B.sha256
# penis-lora-front-view-blowjob-cumshot-taz-wan-21-14b-13b-t2v-and-i2v.safetensors  pov-missionary-t2v_v1_1_14B.sha256

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
  rm -rf /home/comfyuser/ComfyUI/models/style_models && ln -s /workspace/models/style_models /home/comfyuser/ComfyUI/models/style_models
  rm -rf /home/comfyuser/ComfyUI/models/SEEDVR2 && ln -s /workspace/models/SEEDVR2 /home/comfyuser/ComfyUI/models/SEEDVR2
  rm -rf /home/comfyuser/ComfyUI/models/salient && ln -s /workspace/models/salient /home/comfyuser/ComfyUI/models/salient
  rm -rf /home/comfyuser/ComfyUI/models/grounding-dino && ln -s /workspace/models/grounding-dino /home/comfyuser/ComfyUI/models/grounding-dino
  rm -rf /home/comfyuser/ComfyUI/user && ln -s /workspace/user /home/comfyuser/ComfyUI/user
  rm -rf /home/comfyuser/OneTrainerConfigs && ln -s /workspace/OneTrainer /home/comfyuser/OneTrainerConfigs
  rm -rf /home/comfyuser/MusubiConfigs && ln -s /workspace/Musubi /home/comfyuser/MusubiConfigs
  rm -rf /home/comfyuser/OstrisTrainer && ln -s /workspace/OstrisTrainer /home/comfyuser/OstrisTrainer
  rm -rf /home/comfyuser/loras && ln -s /workspace/models/loras /home/comfyuser/loras
  rm -rf /home/comfyuser/ComfyUI/models/mmaudio && ln -s /workspace/models/mmaudio /home/comfyuser/ComfyUI/models/mmaudio
  rm -rf /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger/models && ln -s /workspace/models/ComfyUI-WD14-Tagger/models /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger/models
  rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts && ln -s /workspace/models/comfyui_controlnet_aux/ckpts /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts
  rm -rf /home/comfyuser/ComfyUI/models/bagel && ln -s /workspace/models/bagel /home/comfyuser/ComfyUI/models/bagel
  rm -rf /home/comfyuser/ComfyUI/models/detection && ln -s /workspace/models/detection /home/comfyuser/ComfyUI/models/detection
  rm -rf /home/comfyuser/ComfyUI/models/hyperswap && ln -s /workspace/models/hyperswap /home/comfyuser/ComfyUI/models/hyperswap
fi

# Start cloudflared in the background
# start_cloudflared() {
#     if [ -n "$TUNNEL_TOKEN" ] && [ "$TUNNEL_TOKEN" != "SET_YOUR_TUNNEL_TOKEN" ]; then
#       echo "Starting cloudflared..."
#       echo "Using tunnel token and name..."
#       cloudflared tunnel \
#           --loglevel info \
#           --logfile /var/log/cloudflared.log \
#           run --token "$TUNNEL_TOKEN" "$TUNNEL_NAME"
#     else
#       echo "Skipping private cloudflared, tunnel data not specified"
#       cloudflared tunnel --url https://127.0.0.1 --loglevel info
#     fi
# } 

# Start ComfyUI
start_comfyui() {
    # Capture additional arguments from environment variables
    echo "Starting ComfyUI..."
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1 | cut -d'.' -f1)

    if [ -n "$ROAMING_WAN" ] && [ "$ROAMING_WAN" = "1" ]; then
      python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    elif [ -n "$DEBUG" ] && [ "$DEBUG" = "1" ] && [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ] && [ "$DRIVER_VERSION" -ge 580 ]; then
      /workspace/venv_cc12_cuda130/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
      /workspace/venv_cc12_cuda129/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation
    elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "CPU" ]; then
      /workspace/venv_cc12_cuda129/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --enable-cors-header "*" --cpu
    fi

    #/workspace/venv_cc12_cuda129/bin/python -m pip install --force-reinstall --no-deps protobuf==4.25.3 numpy==1.26.4
}

# Start ComfyUI MultiGPU
start_comfyui_multigpu() {
    if [ -n "$MULTIGPU" ] && [ "$MULTIGPU" = "1" ]; then
      echo "Starting ComfyUI 2..."
      if [ -n "$ROAMING_WAN" ] && [ "$ROAMING_WAN" = "1" ]; then
        python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation --port 8288 --cuda-device 1
      elif [ -n "$CC_VERSION" ] && [ "$CC_VERSION" = "12" ]; then
        /workspace/venv_cc12_cuda129/bin/python /home/comfyuser/ComfyUI/main.py --max-upload-size 300 --dont-print-server --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers --fast fp16_accumulation --port 8288 --cuda-device 1
      fi
    fi
}
#
start_jupyterlab() {
    echo "Starting Jupyter Lab..."
    jupyter lab --allow-root --no-browser --port=8888 --ip=* --FileContentsManager.delete_to_trash=False --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=*
}

start_nginx() {
  echo "Starting Nginx..."
  nginx -g "daemon off;"
}

start_lora_viewer() {
    echo "Starting Lora Viewer..."
    # Start the training UI
    /workspace/venv_onetrainer/bin/python /home/comfyuser/lora-metadata-viewer-fork/app.py --directory /workspace/models/loras
}

# start_cloudflared &
start_comfyui &
start_comfyui_multigpu &
start_lora_viewer &
start_nginx &
start_jupyterlab