#!/bin/bash
set -x
mkdir -p /root/.config/rclone

cat << EOF > /root/.config/rclone/rclone.conf
[r2]
type = s3
provider = Cloudflare
access_key_id = $R2_ID
secret_access_key = $R2_KEY
endpoint = https://c45a8b257df7bbf1fb0c4fb396042547.r2.cloudflarestorage.com
acl = private

[google-drive]
type = drive
client_id =
client_secret =
scope = drive
token = $GOOGLE_DRIVE_TOKEN_JSON
EOF
# TODO: FINISH GOOGLE DRIVE SERVICE ACCOUNT LOGIC

# client_id = YOUR_GOOGLE_DRIVE_CLIENT_ID
# client_secret = YOUR_GOOGLE_DRIVE_CLIENT_SECRET

cp /home/comfyuser/docker/nginx/nginx.conf /etc/nginx/nginx.conf
cp /home/comfyuser/docker/nginx/site-conf/default.conf /etc/nginx/conf.d/default.conf

rclone mount --daemon --log-level=DEBUG --cache-dir=/mnt/r2_cache --allow-other --vfs-cache-max-age=12h --dir-cache-time=12h --poll-interval=10s --vfs-cache-mode=full --vfs-read-chunk-size=128M --fast-list --buffer-size=64M --use-server-modtime r2:europe-colab/models /home/comfyuser/r2_bucket 

rm -rf /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/reactor_sfw.py
rclone copy --log-level=INFO r2:europe-colab/reactor_sfw.py /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node/scripts/

# reactor breaks loading often
mv /home/comfyuser/ComfyUI/custom_nodes/comfyui-reactor-node /tmp/comfyui-reactor-node

rm -rf /home/comfyuser/ComfyUI/models/vae_approx
rclone copy --log-level=INFO r2:europe-colab/extra_model_paths.yaml /home/comfyuser/ComfyUI/
ln -s /home/comfyuser/r2_bucket/models/vae_approx /home/comfyuser/ComfyUI/models/vae_approx
ln -s /home/comfyuser/r2_bucket/models/facerestore_models /home/comfyuser/ComfyUI/models/facerestore_models
ln -s /home/comfyuser/r2_bucket/models/facedetection /home/comfyuser/ComfyUI/models/facedetection
ln -s /home/comfyuser/r2_bucket/models/liveportrait /home/comfyuser/ComfyUI/models/liveportrait
ln -s /home/comfyuser/r2_bucket/models/reactor /home/comfyuser/ComfyUI/models/reactor
ln -s /home/comfyuser/r2_bucket/models/xlabs /home/comfyuser/ComfyUI/models/xlabs
ln -s /home/comfyuser/r2_bucket/models/BiRefNet /home/comfyuser/ComfyUI/models/BiRefNet
ln -s /home/comfyuser/r2_bucket/models/onnx /home/comfyuser/ComfyUI/models/onnx
ln -s /home/comfyuser/r2_bucket/models/LLM /home/comfyuser/ComfyUI/models/LLM
ln -s /home/comfyuser/r2_bucket/models/insightface /home/comfyuser/ComfyUI/models/insightface
ln -s /home/comfyuser/r2_bucket/models/ipadapter /home/comfyuser/ComfyUI/models/ipadapter
ln -s /home/comfyuser/r2_bucket/models/ultralytics /home/comfyuser/ComfyUI/models/ultralytics

# Start cloudflared in the background
start_cloudflared() {
    echo "Starting cloudflared..."
    echo "Using tunnel token and name..."
    cloudflared tunnel \
        --loglevel debug \
        --logfile /var/log/cloudflared.log \
        run --token "$TUNNEL_TOKEN" "$TUNNEL_NAME"
}

# Start ComfyUI
start_comfyui() {
    echo "Starting ComfyUI..."
    if [ -n "$COMMAND" ]; then
      exec sh -c "$COMMAND"
    else
      python ComfyUI/main.py --max-upload-size 100 --dont-print-server --preview-method taesd --enable-cors-header "*" --use-pytorch-cross-attention --disable-xformers
    fi
}

start_nginx() {
  echo "Starting Nginx..."
  nginx -g daemon off
}

start_cloudflared &
start_comfyui &
start_nginx