from huggingface_hub import snapshot_download
import os
HF_HOME = os.getenv("HF_HOME")
if HF_HOME is None:
    HF_HOME = "/home/jinzhenxiong/pretrain"

HF_ENDPOINT="https://hf-mirror.com"


snapshot_download(
    "madebyollin/sdxl-vae-fp16-fix",
    local_dir="/home/jinzhenxiong/pretrain"
)