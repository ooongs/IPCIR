from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("/data/jinzhenxiong/temp/stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "naked woman in a room with a bed"

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

image.save("sdxl.jpg")