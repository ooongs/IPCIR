import json
import time
import traceback

from tqdm import tqdm
from openai import AsyncOpenAI
import asyncio

import argparse

parser = argparse.ArgumentParser(description="Self-consistent reasoning and editing.")
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

BLIP2_MODEL = 'opt' # or 'opt' or 't5'
NUM_CAPTION = 15
with open(input_json, "r") as f:
    annotations = json.load(f)

api_client = API_client(rate_limit=NUM_CAPTION)

async def process_captions(ans):
    # provide a safe default for rel_cap so static analyzers don't complain
    rel_cap = ans.get("relative_caption")
    if DATASET == 'circo':
        rel_cap = ans["relative_caption"]
    elif DATASET == 'circo_val':
        rel_cap = ans["relative_caption"]
    blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
    gpt_answers = ans.get("multi_gpt-3.5_{}".format(BLIP2_MODEL))
    sc_ans = ""
    # Build readable lists for the model
    caps_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(blip2_caption)]) if blip2_caption else "(no multi captions available)"
    gpt_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(gpt_answers)]) if gpt_answers else "(no candidate edits available)"

    task_desc = "Your task is to generate an edited image description based on the given instruction and image caption."

    user_prompt = (
        "Now, based on the following Instruction, Multi Image Captions, and GPT Candidates, \n"
        f"Instruction: {rel_cap}\n\n"
        f"Multi Image Captions:\n{caps_text}\n\n"
        f"GPT Candidates (edited descriptions generated from each caption):\n{gpt_text}\n\n"
        "Please: choose, merge or rewrite the most accurate edited image description based on the Instruction, the Multi Image Captions and the GPT Candidates. "
        "Output exactly one line that begins with 'Edited Description:' and contains only a concise description of the edited image (no extra explanation).\n"
        "Edited Description:"
    )

    # Provide a short example to show format
    messages = [
        {"role": "user", "content": f"{task_desc}\n\nInstruction: has the woman and the man with the roles switched.\nImage Caption: a man adjusting a woman's tie.\nEdited Description:"},
        {"role": "assistant", "content": "a woman adjusting a man's tie."},
        {"role": "user", "content": user_prompt},
    ]

    response = await api_client(
        model=model_name,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
    )

    sc_ans = response.strip()
    if sc_ans.startswith("Edited Description:"):
        sc_ans = sc_ans[len("Edited Description:"):].strip()

    new_ans = copy.deepcopy(ans)
    new_ans["self_consistent_{}".format(BLIP2_MODEL)] = sc_ans
    return new_ans

async def main():
    new_annotations = []
    batch_size = 25
    batches = [annotations[i:i + batch_size] for i in range(0, len(annotations), batch_size)]
    for batch in tqdm(batches):
        async def process_batch(batch):
            batch_tasks = []
            for idx, ans in enumerate(batch):
                batch_tasks.append(process_captions(ans))
            results = await asyncio.gather(*batch_tasks)
            return results
        batch_results = await process_batch(batch)
        new_annotations.extend(batch_results)

    return new_annotations

if __name__ == "__main__":
    new_annotations = asyncio.run(main())
    # 将处理后的结果写入新文件，避免覆盖原始数据
    out_path = input_json + '.SC.json'
    with open(out_path, "w") as f:
        f.write(json.dumps(new_annotations, indent=4))