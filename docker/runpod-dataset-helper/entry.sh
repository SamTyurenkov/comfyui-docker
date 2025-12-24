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

export PATH="/workspace/venv_onetrainer/bin:$PATH"

which python
which pip

cp /home/comfyuser/docker/nginx/nginx.conf /etc/nginx/nginx.conf
cp /home/comfyuser/docker/nginx/site-conf/default.conf /etc/nginx/conf.d/default.conf

rm -rf /home/comfyuser/OneTrainer && ln -s /workspace/OneTrainer /home/comfyuser/OneTrainer
rm -rf /home/comfyuser/wd14_tagger/models && ln -s /workspace/models/ComfyUI-WD14-Tagger/models /home/comfyuser/wd14_tagger/models
rm -rf /home/comfyuser/loras && ln -s /workspace/models/loras /home/comfyuser/loras

start_jupyterlab() {
    echo "Starting Jupyter Lab..."
    jupyter lab --allow-root --no-browser --port=8888 --ip=* --FileContentsManager.delete_to_trash=False --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=*
}

# start_tensorboard() {
#     echo "Starting TensorBoard..."
#     tensorboard --logdir=/home/comfyuser/OneTrainer/tensorboard
# }

start_dataset_ui() {
    echo "Starting Training UI..."
    # Start the training UI
    /workspace/venv_onetrainer/bin/python /home/comfyuser/dataset_ui.py
}

start_nginx() {
  echo "Starting Nginx..."
  nginx -g "daemon off;"
}

# start_tensorboard &
start_dataset_ui &
start_nginx &
start_jupyterlab
