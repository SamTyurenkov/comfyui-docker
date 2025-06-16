# syntax=docker/dockerfile:1
#see https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds
FROM python:3.12 as builder

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
EXPOSE 8888

RUN mkdir -p /home/comfyuser/docker/
COPY site-conf/ /home/comfyuser/docker/nginx/site-conf
COPY nginx.conf /home/comfyuser/docker/nginx/nginx.conf
COPY ssl/self-signed/cert.pem /home/comfyuser/docker/nginx/ssl/cert.pem
COPY ssl/self-signed/privkey.pem /home/comfyuser/docker/nginx/ssl/privkey.pem


# RUN python -m venv --copies /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

RUN git clone https://github.com/comfyanonymous/ComfyUI /home/comfyuser/ComfyUI

COPY extra_model_paths.yaml /home/comfyuser/ComfyUI/extra_model_paths.yaml

# Copied from WAN Colab
# RUN git clone https://github.com/thu-ml/SageAttention.git /workspace/sageattention
RUN git clone https://github.com/shinich39/comfyui-get-meta /home/comfyuser/ComfyUI/custom_nodes/comfyui-get-meta
RUN git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_JPS-Nodes
RUN git clone https://github.com/aria1th/ComfyUI-LogicUtils /home/comfyuser/ComfyUI/custom_nodes/comfyui-logicutils
RUN git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git /home/comfyuser/ComfyUI/custom_nodes/teacache
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui /home/comfyuser/ComfyUI/custom_nodes/was-node-suite-comfyui
RUN git clone https://github.com/bash-j/mikey_nodes /home/comfyuser/ComfyUI/custom_nodes/mikey_nodes

# Copied from Generic Colab
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager /home/comfyuser/ComfyUI/custom_nodes/comfyui-manager
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-manager && git checkout 3.32.5
RUN git clone https://github.com/chrisgoringe/cg-use-everywhere /home/comfyuser/ComfyUI/custom_nodes/cg-use-everywhere
RUN cd /home/comfyuser/ComfyUI/custom_nodes/cg-use-everywhere && git checkout 5ae64f462ec24fc0100a309c42c3def6aae18d7b
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux && git checkout 7c4f6fb6ff18aa6dbfd4f77c2e8ad46119b5d84d
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus /home/comfyuser/ComfyUI/custom_nodes/comfyui_ipadapter_plus
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui_ipadapter_plus && git checkout a0f451a5113cf9becb0847b92884cb10cbdec0ef
RUN git clone https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg /home/comfyuser/ComfyUI/custom_nodes/comfyui-inspyrenet-rembg
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-inspyrenet-rembg && git checkout 87ac452ef1182e8f35f59b04010158d74dcefd06
RUN git clone https://github.com/Gourieff/ComfyUI-ReActor /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node && git checkout v0.6.0
RUN git clone https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch /home/comfyuser/ComfyUI/custom_nodes/comfyui-inpaint-cropandstitch
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack && git checkout 8.15.1
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-subpack
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-subpack && git checkout 1.3.2
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes
RUN git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait /home/comfyuser/ComfyUI/custom_nodes/comfyui-advancedliveportrait
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-advancedliveportrait && git checkout 3bba732915e22f18af0d221b9c5c282990181f1b
RUN git clone https://github.com/XLabs-AI/x-flux-comfyui /home/comfyuser/ComfyUI/custom_nodes/x-flux-comfyui
RUN git clone https://github.com/crystian/ComfyUI-Crystools /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Crystools
RUN cd /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Crystools && git checkout 1156ff983b635ef506e7b79659126837a1e9d275
RUN git clone https://github.com/twri/sdxl_prompt_styler /home/comfyuser/ComfyUI/custom_nodes/sdxl_prompt_styler
RUN git clone https://github.com/rgthree/rgthree-comfy /home/comfyuser/ComfyUI/custom_nodes/rgthree-comfy
RUN cd /home/comfyuser/ComfyUI/custom_nodes/rgthree-comfy && git checkout 2fcd7a64217741ec11fe9429d15200e5477bc19a
RUN git clone https://github.com/cubiq/ComfyUI_essentials /home/comfyuser/ComfyUI/custom_nodes/comfyui_essentials
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui_essentials && git checkout 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9
RUN git clone https://github.com/kijai/ComfyUI-LBMWrapper /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-LBMWrapper
RUN cd /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-LBMWrapper && git checkout 6bda0e7c6910033f2efc124131422fae568965db
RUN git clone https://github.com/sipherxyz/comfyui-art-venture /home/comfyuser/ComfyUI/custom_nodes/comfyui-art-venture
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-art-venture && git checkout c3bacdc0c4393166b25c5a8f30a183292d0010e8
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite && git checkout a7ce59e381934733bfae03b1be029756d6ce936d
RUN git clone https://github.com/SamTyurenkov/comfyui_chatgpt /home/comfyuser/ComfyUI/custom_nodes/comfyui_chatgpt
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use /home/comfyuser/ComfyUI/custom_nodes/comfyui-easy-use
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfyui-easy-use && git checkout 71c7865d2d3c934ccb99f24171e08ae5a81148ac
RUN git clone https://github.com/ManglerFTW/ComfyI2I /home/comfyuser/ComfyUI/custom_nodes/ComfyI2I
RUN git clone https://github.com/melMass/comfy_mtb /home/comfyuser/ComfyUI/custom_nodes/comfy_mtb
RUN cd /home/comfyuser/ComfyUI/custom_nodes/comfy_mtb && git checkout 4574c6451c1d7c256cdd062a01eff95099caf99f
RUN git clone https://github.com/Pfaeff/pfaeff-comfyui /home/comfyuser/ComfyUI/custom_nodes/pfaeff-comfyui
RUN git clone https://github.com/jags111/efficiency-nodes-comfyui /home/comfyuser/ComfyUI/custom_nodes/efficiency-nodes-comfyui
RUN cd /home/comfyuser/ComfyUI/custom_nodes/efficiency-nodes-comfyui && git checkout 7dc3c93aa27fb200d62179b9a9f2b4fe301b4474
RUN git clone https://github.com/M1kep/ComfyLiterals /home/comfyuser/ComfyUI/custom_nodes/ComfyLiterals
RUN git clone https://github.com/kijai/ComfyUI-KJNodes /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-KJNodes
RUN cd /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-KJNodes && git checkout a879d152083ebcbce90d187b5537193034974a54
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation
RUN git clone https://github.com/kijai/ComfyUI-SUPIR /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SUPIR
RUN git clone https://github.com/cdb-boop/ComfyUI-Bringing-Old-Photos-Back-to-Life /home/comfyuser/ComfyUI/custom_nodes/bringing-old-photos-back-to-life
RUN git clone https://github.com/cdb-boop/comfyui-image-round /home/comfyuser/ComfyUI/custom_nodes/comfyui-image-round
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts /home/comfyuser/ComfyUI/custom_nodes/comfyui-custom-scripts
RUN git clone https://github.com/Zehong-Ma/ComfyUI-MagCache /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-MagCache
RUN git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale /home/comfyuser/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper
RUN git clone https://github.com/neverbiasu/ComfyUI-SAM2 /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-SAM2
RUN git clone https://github.com/pythongosssss/ComfyUI-WD14-Tagger /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-WD14-Tagger
RUN git clone https://github.com/Eagle-CN/ComfyUI-Addoor /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Addoor


RUN wget -O /tmp/cloudflared-linux-amd64.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
RUN dpkg -i /tmp/cloudflared-linux-amd64.deb

ADD entry.sh /usr/local/bin/entrypoint.sh
WORKDIR /home/comfyuser/
RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]