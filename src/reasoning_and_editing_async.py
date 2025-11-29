import json
import time
import traceback

from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")
parser.add_argument('--dress',default='toptee',type=str)
parser.add_argument('--model_name', default='Qwen/Qwen1.5-32B-Chat', type=str)
args = parser.parse_args()

model_name = args.model_name

# deep copy for safe non-destructive updates
import copy

# vllm.__version__ = "0.6.3.post1"
# vllm serve Qwen/Qwen1.5-32B-Chat --dtype auto --port 8000 --trust-remote-code --max_model_len 4096
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
class API_client():
    def __init__(self, rate_limit) -> None:
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(self.rate_limit)
    async def __call__(self, **kwargs):
        async with self.semaphore:
            completion = await client.chat.completions.create(
                **kwargs
            )
            response = completion.choices[0].message.content
        return response

DATASET = 'circo' # 'cirr', 'fashioniq'
# DATASET = 'cirr_val'

input_json = ''

if DATASET == 'circo':
    SPLIT = 'test'
    input_json = 'CIRCO/annotations/test.json'
elif DATASET == 'circo_val':
    SPLIT = 'val'
    input_json = 'CIRCO/annotations/val.json'
elif DATASET == 'cirr':
    SPLIT = 'test1'
    input_json = 'CIRR/cirr/captions/cap.rc2.test1.json'
elif DATASET == 'cirr_val':
    SPLIT = 'val'
    input_json = 'CIRR/cirr/captions/cap.rc2.val.json'
elif DATASET == 'fashioniq':
    SPLIT = 'val'
    DRESS = args.dress # 'shirt', 'toptee', 'dress
    new_annotations = []
    input_json = f'/mnt/data1/comp_data/fashion-iq/captions/cap.{DRESS}.{SPLIT}.json'

BLIP2_MODEL = 'opt' # or 'opt' or 't5'
MULTI_CAPTION = True
NUM_CAPTION = 15
with open(input_json, "r") as f:
    annotations = json.load(f)

api_client = API_client(rate_limit=NUM_CAPTION)

async def process_annotation(ans, idx):
    if DATASET == 'circo':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'circo_val':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'cirr':
        rel_cap = ans["caption"]
    elif DATASET == 'cirr_val':
        rel_cap = ans["caption"]
    else:
        rel_cap = ans['captions'][0]

    if MULTI_CAPTION:
        blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
    else:
        if BLIP2_MODEL == 'none':
            blip2_caption = ans["shared_concept"]
        elif BLIP2_MODEL == 'opt':
            blip2_caption = ans["blip2_caption"]
        else:
            blip2_caption = ans["blip2_caption_{}".format(BLIP2_MODEL)]

    sys_prompt = """
        I have an image. Given an instruction to edit the image, carefully generate a description of the edited image. I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". 
        The edited description you generate should begin with \"Edited Description:\". "
        You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.
    """
    if MULTI_CAPTION:
        # 在单次 process 中并发发送该注释下所有 blip2_caption 的请求，然后一次性收集返回
        async def call_cap(cap):
            while True:
                try:
                    messages = [
                        {"role": "system", "content": sys_prompt}
                    ]
                    messages.append({"role": "user", "content": "Image Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\n"})
                    messages.append({"role": "assistant", "content": "Edited Description: a woman adjusting a man's tie."})
                    messages.append({"role": "user", "content": "Image Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)})
                    response = await api_client(
                        model=model_name,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.0,
                    )
                    return response.strip()
                except Exception:
                    traceback.print_exc()
                    await asyncio.sleep(3)

        # 为当前 annotation 中的每个 cap 创建任务并并发执行
        cap_tasks = [call_cap(cap) for cap in blip2_caption]
        multi_gpt = await asyncio.gather(*cap_tasks)

        # 返回一个新的 annotation 副本，避免并发修改原始 list 中的对象
        new_ans = copy.deepcopy(ans)
        new_ans["multi_gpt-3.5_{}".format(BLIP2_MODEL)] = multi_gpt
        return idx, new_ans
    else:
        # 未开启 MULTI_CAPTION 时也返回深拷贝，保持不在原地修改的语义
        new_ans = copy.deepcopy(ans)
        return idx, new_ans

async def main():
    # 顺序执行每个 annotation 的处理，便于使用 tqdm 显示进度
    new_annotations = []
    for i, ans in enumerate(tqdm(annotations, desc="Processing annotations")):
        _, new_ans = await process_annotation(ans, i)
        new_annotations.append(new_ans)

    return new_annotations

if __name__ == "__main__":
    new_annotations = asyncio.run(main())
    # 将处理后的结果写入新文件，避免覆盖原始数据
    out_path = input_json + '.processed.json'
    with open(out_path, "w") as f:
        f.write(json.dumps(new_annotations, indent=4))