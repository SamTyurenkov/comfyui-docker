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
    git

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio torchsde --extra-index-url https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/pip pip install diffusers aiohttp aiodns Brotli numpy==2.0 onnxruntime-gpu flet matplotlib-inline albumentations spandrel kornia av transparent-background xformers insightface

COPY ComfyUI/ /home/comfyuser/ComfyUI/

RUN --mount=type=cache,target=/root/.cache/pip pip install comfyui-workflow-templates -r /home/comfyuser/ComfyUI/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui_controlnet_aux/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-pack/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-impact-subpack/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-advancedliveportrait/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-LBMWrapper/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-art-venture/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-videohelpersuite/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui_chatgpt/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/comfyui-easy-use/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /home/comfyuser/ComfyUI/custom_nodes/ComfyI2I/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python /home/comfyuser/ComfyUI/custom_nodes/x-flux-comfyui/setup.py

RUN rm -rf /home/comfyuser/ComfyUI/models && rm -rf /home/comfyuser/ComfyUI/user && rm -rf /home/comfyuser/ComfyUI/output
WORKDIR /home/comfyuser/

# FROM python:3.12-slim as runtime
# COPY --from=builder /home/comfyuser /home/comfyuser
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y rclone fuse libfuse2

ADD entry.sh /usr/local/bin/entrypoint.sh
RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
WORKDIR /home/comfyuser/