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
    libgl1 \
    cmake \
    nginx \
    git

# Nginx folders
RUN mkdir -p /var/cache/nginx/client_temp \
	&& mkdir -p /var/cache/nginx/proxy_temp \
	&& mkdir -p /var/cache/nginx/fastcgi_temp \
	&& mkdir -p /var/cache/nginx/uwsgi_temp \
	&& mkdir -p /var/cache/nginx/scgi_temp

# Expose ports
EXPOSE 80 81
EXPOSE 6006
EXPOSE 5000

COPY site-conf/ /home/comfyuser/docker/nginx/site-conf
COPY nginx.conf /home/comfyuser/docker/nginx/nginx.conf
COPY ssl/self-signed/cert.pem /home/comfyuser/docker/nginx/ssl/cert.pem
COPY ssl/self-signed/privkey.pem /home/comfyuser/docker/nginx/ssl/privkey.pem

RUN mkdir -p /home/comfyuser/OneTrainer/

RUN cd /home/comfyuser && git clone https://github.com/Nerogar/OneTrainer.git

RUN pip install --upgrade pip && pip install jupyter virtualenv flask flask-cors

# Copy training UI files
COPY training_ui.py /home/comfyuser/training_ui.py
COPY templates/ /home/comfyuser/templates/
COPY find_configs.py /home/comfyuser/find_configs.py

ADD entry.sh /usr/local/bin/entrypoint.sh
WORKDIR /home/comfyuser/


# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
# RUN dpkg -i cuda-keyring_1.1-1_all.deb
# RUN apt-get update && apt-get -y install cuda-toolkit-12-9
# ENV CUDA_HOME="/usr/local/cuda-12.9"
# ENV PATH="$CUDA_HOME/bin:$PATH"
# ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64"

RUN ["chmod", "+x", "/usr/local/bin/entrypoint.sh"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
