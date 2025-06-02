#!/bin/bash
# set +x
# echo "$R2_ID:$R2_KEY" > s3fs.passwd
# set -x
# chmod 600 s3fs.passwd
# rm -rf /home/comfyuser/ComfyUI/models
# #Mount the bucket using s3fs
# s3fs colab-models /home/comfyuser/ComfyUI/models -o parallel_count='64' -o multipart_size="512" -o use_cache='/mnt/r2_cache' -o passwd_file='s3fs.passwd' -o url=https://c45a8b257df7bbf1fb0c4fb396042547.r2.cloudflarestorage.com -o use_path_request_style,url=https://c45a8b257df7bbf1fb0c4fb396042547.r2.cloudflarestorage.com -d
# echo "Mounted r2_bucket"
# rm 's3fs.passwd'
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
EOF

rm -rf /home/comfyuser/ComfyUI/models
mkdir -p /home/comfyuser/ComfyUI/models
rclone mount --daemon --log-level=DEBUG --cache-dir=/mnt/r2_cache --allow-other --vfs-cache-max-age=12h --dir-cache-time=12h --poll-interval=10s --vfs-cache-mode=full --vfs-read-chunk-size=128M --fast-list --buffer-size=64M --use-server-modtime r2:colab-models/models /home/comfyuser/ComfyUI/models 
# rclone mount --log-level=DEBUG --cache-dir=/mnt/r2_cache --allow-other --vfs-cache-max-age=12h --dir-cache-time=12h --poll-interval=10s --vfs-cache-mode=full --vfs-read-chunk-size=128M --fast-list --buffer-size=64M --use-server-modtime r2:colab-models/models /home/comfyuser/ComfyUI/models

# # Execute the main command
if [ -n "$COMMAND" ]; then
  exec sh -c "$COMMAND"
fi