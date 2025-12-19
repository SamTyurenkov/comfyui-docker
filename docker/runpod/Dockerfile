# syntax=docker/dockerfile:1
#see https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds
FROM python:3.12-bookworm as builder

# Update package lists and install dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libssl3 \
    wget \
    cmake \
    git \
    lsb-release \
    libssl-dev \
    ca-certificates \
    curl \
    openssl \
    cron \
    rclone \
    awscli \
    aria2 \
    ffmpeg \
    # fuse \
    # libfuse2 \
    nginx


# Nginx folders
RUN mkdir -p /var/cache/nginx/client_temp \
	&& mkdir -p /var/cache/nginx/proxy_temp \
	&& mkdir -p /var/cache/nginx/fastcgi_temp \
	&& mkdir -p /var/cache/nginx/uwsgi_temp \
	&& mkdir -p /var/cache/nginx/scgi_temp

# Expose ports
EXPOSE 80 443
EXPOSE 8888 8889

RUN mkdir -p /home/comfyuser/docker/
COPY site-conf/ /home/comfyuser/docker/nginx/site-conf
COPY nginx.conf /home/comfyuser/docker/nginx/nginx.conf
COPY ssl/self-signed/cert.pem /home/comfyuser/docker/nginx/ssl/cert.pem
COPY ssl/self-signed/privkey.pem /home/comfyuser/docker/nginx/ssl/privkey.pem


# RUN python -m venv --copies /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /home/comfyuser/ComfyUI

COPY extra_model_paths.yaml /home/comfyuser/ComfyUI/extra_model_paths.yaml

# Copied from WAN Colab
# RUN git clone https://github.com/thu-ml/SageAttention.git /workspace/sageattention
RUN git clone --depth 1 https://github.com/shinich39/comfyui-get-meta /home/comfyuser/ComfyUI/custom_nodes/comfyui-get-meta
RUN git clone --depth 1 https://github.com/JPS-GER/ComfyUI_JPS-Nodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_JPS-Nodes
RUN git clone --depth 1 https://github.com/aria1th/ComfyUI-LogicUtils /home/comfyuser/ComfyUI/custom_nodes/comfyui-logicutils
RUN git clone --depth 1 https://github.com/welltop-cn/ComfyUI-TeaCache.git /home/comfyuser/ComfyUI/custom_nodes/teacache
RUN git clone --depth 1 https://github.com/WASasquatch/was-node-suite-comfyui /home/comfyuser/ComfyUI/custom_nodes/was-node-suite-comfyui
RUN git clone --depth 1 https://github.com/bash-j/mikey_nodes /home/comfyuser/ComfyUI/custom_nodes/mikey_nodes

# Copied from Generic Colab
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager /home/comfyuser/ComfyUI/custom_nodes/comfyui-manager
RUN git clone --depth 1 https://github.com/chrisgoringe/cg-use-everywhere /home/comfyuser/ComfyUI/custom_nodes/cg-use-everywhere
RUN git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus /home/comfyuser/ComfyUI/custom_nodes/comfyui_ipadapter_plus
RUN git clone --depth 1 https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg /home/comfyuser/ComfyUI/custom_nodes/comfyui-inspyrenet-rembg
RUN git clone https://github.com/Gourieff/ComfyUI-ReActor /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node
RUN git clone --depth 1 https://github.com/SamTyurenkov/ComfyUI-Inpaint-CropAndStitch /home/comfyuser/ComfyUI/custom_nodes/comfyui-inpaint-cropandstitch
RUN git clone https://github.com/SamTyurenkov/ComfyUI-Impact-Pack /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack && git checkout preview-bridge-video
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Subpack /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-subpack
RUN git clone --depth 1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes
RUN git clone --depth 1 https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait /home/comfyuser/ComfyUI/custom_nodes/comfyui-advancedliveportrait
RUN git clone --depth 1 https://github.com/XLabs-AI/x-flux-comfyui /home/comfyuser/ComfyUI/custom_nodes/x-flux-comfyui
RUN git clone --depth 1 https://github.com/crystian/ComfyUI-Crystools /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Crystools
RUN git clone --depth 1 https://github.com/twri/sdxl_prompt_styler /home/comfyuser/ComfyUI/custom_nodes/sdxl_prompt_styler
RUN git clone --depth 1 https://github.com/rgthree/rgthree-comfy /home/comfyuser/ComfyUI/custom_nodes/rgthree-comfy
RUN git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials /home/comfyuser/ComfyUI/custom_nodes/comfyui_essentials
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-LBMWrapper /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-LBMWrapper
RUN git clone --depth 1 https://github.com/sipherxyz/comfyui-art-venture /home/comfyuser/ComfyUI/custom_nodes/comfyui-art-venture
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite
# RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite && git checkout a7ce59e381934733bfae03b1be029756d6ce936d
RUN git clone --depth 1 https://github.com/SamTyurenkov/comfyui_chatgpt /home/comfyuser/ComfyUI/custom_nodes/comfyui_chatgpt
RUN git clone --depth 1 https://github.com/SamTyurenkov/comfyui-vace-preprocessors /home/comfyuser/ComfyUI/custom_nodes/comfyui-vace-preprocessors
RUN git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use /home/comfyuser/ComfyUI/custom_nodes/comfyui-easy-use
RUN git clone --depth 1 https://github.com/ManglerFTW/ComfyI2I /home/comfyuser/ComfyUI/custom_nodes/ComfyI2I
RUN git clone --depth 1 https://github.com/melMass/comfy_mtb /home/comfyuser/ComfyUI/custom_nodes/comfy_mtb
RUN git clone --depth 1 https://github.com/Pfaeff/pfaeff-comfyui /home/comfyuser/ComfyUI/custom_nodes/pfaeff-comfyui
RUN git clone --depth 1 https://github.com/jags111/efficiency-nodes-comfyui /home/comfyuser/ComfyUI/custom_nodes/efficiency-nodes-comfyui
RUN git clone --depth 1 https://github.com/M1kep/ComfyLiterals /home/comfyuser/ComfyUI/custom_nodes/ComfyLiterals
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-KJNodes
# RUN cd /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-KJNodes && git checkout a879d152083ebcbce90d187b5537193034974a54
RUN git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-SUPIR /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SUPIR
RUN git clone --depth 1 https://github.com/cdb-boop/ComfyUI-Bringing-Old-Photos-Back-to-Life /home/comfyuser/ComfyUI/custom_nodes/bringing-old-photos-back-to-life
RUN git clone --depth 1 https://github.com/cdb-boop/comfyui-image-round /home/comfyuser/ComfyUI/custom_nodes/comfyui-image-round
RUN git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts /home/comfyuser/ComfyUI/custom_nodes/comfyui-custom-scripts
COPY autocomplete.txt /home/comfyuser/ComfyUI/custom_nodes/comfyui-custom-scripts/user/autocomplete.txt
RUN git clone --depth 1 https://github.com/Zehong-Ma/ComfyUI-MagCache /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-MagCache
RUN git clone --depth 1 https://github.com/ssitu/ComfyUI_UltimateSDUpscale /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-HunyuanVideoWrapper /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-HunyuanVideoWrapper
RUN git clone --depth 1 https://github.com/neverbiasu/ComfyUI-SAM2 /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SAM2
RUN git clone --depth 1 https://github.com/pythongosssss/ComfyUI-WD14-Tagger /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger
RUN git clone --depth 1 https://github.com/Eagle-CN/ComfyUI-Addoor /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Addoor
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2 /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Florence2
RUN git clone --depth 1 https://github.com/gameltb/comfyui-stablesr /home/comfyuser/ComfyUI/custom_nodes/comfyui-stablesr
RUN git clone --depth 1 https://github.com/ethansmith2000/comfy-todo /home/comfyuser/ComfyUI/custom_nodes/comfy-todo
RUN git clone --depth 1 https://github.com/kaibioinfo/ComfyUI_AdvancedRefluxControl.git /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_AdvancedRefluxControl
RUN git clone --depth 1 https://github.com/Jonseed/ComfyUI-Detail-Daemon /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Detail-Daemon
# RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-LTXVideo
RUN git clone --depth 1 https://github.com/spacepxl/ComfyUI-Image-Filters /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Image-Filters
RUN git clone --depth 1 https://github.com/Clybius/ComfyUI-ClybsChromaNodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-ClybsChromaNodes
RUN git clone --depth 1 https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler /home/comfyuser/ComfyUI/custom_nodes/sigmas_tools_and_the_golden_scheduler
RUN git clone --depth 1 https://github.com/Nourepide/ComfyUI-Allor /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Allor
RUN git clone --depth 1 https://github.com/chflame163/ComfyUI_LayerStyle /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_LayerStyle
RUN git clone --depth 1 https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-segment-anything-2 /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-segment-anything-2
RUN git clone --depth 1 https://github.com/scraed/LanPaint.git /home/comfyuser/ComfyUI/custom_nodes/LanPaint
RUN git clone --depth 1 https://github.com/1038lab/ComfyUI-JoyCaption /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-JoyCaption
RUN git clone --depth 1 https://github.com/Saquib764/omini-kontext /home/comfyuser/ComfyUI/custom_nodes/omini-kontext
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-MMAudio /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-MMAudio
RUN git clone --depth 1 https://github.com/SamTyurenkov/ComfyUI-BAGEL /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-BAGEL
RUN git clone --depth 1 https://github.com/ShmuelRonen/ComfyUI-VideoUpscale_WithModel /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-VideoUpscale_WithModel
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanAnimatePreprocess /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess
# RUN git clone https://github.com/weilin9999/WeiLin-ComfyUI-prompt-all-in-one /home/comfyuser/ComfyUI/custom_nodes/WeiLin-ComfyUI-prompt-all-in-one
# RUN git clone https://github.com/pollockjj/ComfyUI-MultiGPU /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-MultiGPU
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-GIMM-VFI /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-GIMM-VFI
RUN git clone --depth 1 https://github.com/ChenDarYen/ComfyUI-NAG /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-NAG
RUN git clone --depth 1 https://github.com/SamTyurenkov/ComfyUI-Upscaler-Tensorrt /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Upscaler-Tensorrt
RUN git clone --depth 1 https://github.com/Smirnov75/ComfyUI-mxToolkit /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-mxToolkit
RUN git clone --depth 1 https://github.com/jamesWalker55/comfyui-various /home/comfyuser/comfyui-various
RUN git clone --depth 1 https://github.com/SamTyurenkov/lora-metadata-viewer-fork /home/comfyuser/lora-metadata-viewer-fork

RUN git clone --depth 1 https://github.com/AMTPorn/comfyui_amt /home/comfyuser/ComfyUI/custom_nodes/comfyui_amt


RUN pip install --upgrade pip && pip install jupyter ninja
# RUN wget -O /tmp/cloudflared-linux-amd64.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
# RUN dpkg -i /tmp/cloudflared-linux-amd64.deb
ADD entry.sh /usr/local/bin/entrypoint.sh
WORKDIR /home/comfyuser/
RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
