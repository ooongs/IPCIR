import os


import torch
from diffusers import Step1XEditPipeline
from diffusers.utils import load_image
import json 

pipe = Step1XEditPipeline.from_pretrained("/data/jinzhenxiong/temp/stepfun-ai/Step1X-Edit-v1p1-diffusers", torch_dtype=torch.bfloat16)
pipe.to("cuda")

image_id = 549870
json_file = "/home/jinzhenxiong/Imagine-and-Seek/data/CIRCO/annotations/test.json"
with open(json_file, "r") as f:
    data = json.load(f)
    for item in data:
        if item["reference_img_id"] == image_id:
            prompt = item["relative_caption"]
            break

print("=== processing image ===")
image = load_image(f"/home/jinzhenxiong/Imagine-and-Seek/data/CIRCO/COCO2017_unlabeled/unlabeled2017/000000{image_id}.jpg").convert("RGB")
# prompt = "a person looking at his laptop"
print(prompt)
image.save(f"original_image_{image_id}.jpg")
image = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=20,
    timesteps_truncate=0.5,
    true_cfg_scale=4.0,
    height=1024,
    width=1024,
    guidance_scale=7.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save(f"{image_id}.jpg")

# print(f"Loading model from cache: {CACHE_DIR}")
# pipe = Step1XEditPipelineV1P2.from_pretrained(
#     "/data/jinzhenxiong/temp/models--stepfun-ai--Step1X-Edit", 
#     torch_dtype=torch.bfloat16,
# )
# pipe.to("cuda")
# print("=== processing image ===")
# image = load_image("/home/jinzhenxiong/Imagine-and-Seek/data/CIRCO/COCO2017_unlabeled/unlabeled2017s/000000281438.jpg").convert("RGB")
# prompt = "has a higher quality and is taken during the daytime"
# enable_thinking_mode=True
# enable_reflection_mode=True
# pipe_output = pipe(
#     image=image,
#     prompt=prompt,
#     num_inference_steps=50,
#     true_cfg_scale=6,
#     generator=torch.Generator().manual_seed(42),
#     enable_thinking_mode=enable_thinking_mode,
#     enable_reflection_mode=enable_reflection_mode,
# )
# if enable_thinking_mode:
#     print("Reformat Prompt:", pipe_output.reformat_prompt)
# for image_idx in range(len(pipe_output.images)):
#     pipe_output.images[image_idx].save(f"0001-{image_idx}.jpg", lossless=True)
#     if enable_reflection_mode:
#         print(pipe_output.think_info[image_idx])
#         print(pipe_output.best_info[image_idx])
# pipe_output.final_images[0].save(f"0001-final.jpg", lossless=True)