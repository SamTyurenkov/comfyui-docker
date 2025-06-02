#!/bin/bash
USER_NAME=$(whoami)
# Source the .env file to load environment variables
if [ -f .env ]; then
    . ./.env
fi

docker save -o docker-built/nginx-comfyui.tar nginx-comfyui
docker save -o docker-built/comfyui.tar comfyui

mkdir -p /home/$USER_NAME/.config/rclone
cat << EOF > /home/$USER_NAME/.config/rclone/rclone.conf
[r2]
type = s3
provider = Cloudflare
access_key_id = $R2_ID
secret_access_key = $R2_KEY
endpoint = https://c45a8b257df7bbf1fb0c4fb396042547.r2.cloudflarestorage.com
acl = private
EOF

rclone copy docker-built/ r2:colab-models/docker --verbose