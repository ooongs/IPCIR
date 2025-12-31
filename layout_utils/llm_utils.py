import json
import re
import hashlib
from tqdm import tqdm

from enum_definitions import *
from data_definitions import *
import torch

# from dataset import coco, open_images
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import openai
from openai import OpenAI

class OpenAI_API(object):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def get_response(self, prompt, max_tokens=5, temperature=0.9, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=["\n", " Human:", " AI:"]):
        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            reasoning={ "effort": "minimal" }
        )
        return response

def load_model_llm(args, max_token = 4096, device = 'cuda'):
    # Disable tokenizers parallelism to avoid multiprocessing conflicts
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set torch multiprocessing start method to avoid conflicts
    try:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    tokenizer = None
    model = None

    if args.model_type == 'openai':
        from openai import OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        openai.api_key = api_key
        # Use proxy only if specified
        if os.environ.get('OPENAI_PROXY'):
            openai.proxy = {
                'http': os.environ.get('OPENAI_PROXY'),
                'https': os.environ.get('OPENAI_PROXY')
            }
        model = OpenAI_API(openai.api_key)
    elif args.model_type == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.llm_path, device_map="auto", trust_remote_code=True).eval()
    elif args.model_type == 'deepseek':
        from openai import OpenAI
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        model = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    elif args.model_type == 'qwen-api':
        from openai import OpenAI
        api_key = os.environ.get('QWEN_API_KEY') or os.environ.get('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("QWEN_API_KEY or DASHSCOPE_API_KEY environment variable is not set")
        model = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    return tokenizer, model

def get_chat_prompt(prompt_system, inputs, examples):
    messages = [
        {"role": "system", "content": prompt_system}
    ]
    for sample in examples:
        messages.append({"role": "user", "content": sample["input"]})
        messages.append({"role": "assistant", "content": sample["output"]})
    messages.append({"role": "user", "content": inputs})
    # print(messages)

    return messages

def apply_llm(args, messages, model, tokenizer, max_token = 4096):
    responses = []
    for prompt in messages:
        if args.model_type == 'openai':
            result = model.get_response(prompt, max_tokens=max_token)
            response = result.output_text
            responses.append(response)
            print(response)
        elif args.model_type == 'deepseek':
            response = model.chat.completions.create(
                model="deepseek-chat",
                messages=prompt,
                stream=False
            )
            responses.append(response.choices[0].message.content)
        elif args.model_type == 'qwen':
            device = "cuda" 
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            attention_mask = torch.ones(model_inputs.input_ids.shape,dtype=torch.long,device=device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_token,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        elif args.model_type == 'qwen-api':
            response = model.chat.completions.create(
                model="qwen-flash",
                messages=prompt,
                stream=False
            )
            responses.append(response.choices[0].message.content)
    return responses

def run_llm_inferene(args, model, tokenizer, inputs, prompt_system, examples):
    if isinstance(inputs, str):
        inputs = [inputs]
    prompts = [get_chat_prompt(prompt_system, input_, examples) for input_ in inputs]
    outputs = apply_llm(args, prompts, model, tokenizer)
    return outputs


def llm_layout(args, classes, rule, scene, layout, refer, model, tokenizer, image_idx, dataset_stastic, prompt_system, examples, random = False):
    """Simplified LLM layout generation - only returns label and desc"""
    if random:
        layout_info = generate_layout_random(args, classes, rule, scene, dataset_stastic)
    else:
        # Simple prompt construction
        instruction_input = get_simple_inputs(classes, rule=rule, scene=scene, Initial_Layout=layout, refer=refer)
        basic_input = [get_chat_prompt(prompt_system, instruction_input, examples)]

        # Get LLM response
        reasoning_output = apply_llm(args, basic_input, model, tokenizer)

        # Parse JSON response
        layout_info = parse_json_layout(reasoning_output[0])

        if layout_info is None or len(layout_info) == 0:
            print(f"  → LLM returned invalid or empty layout")
            return False, None, None, None

    messages_hash = hashlib.sha256(json.dumps(layout_info, sort_keys=True).encode('utf-8')).hexdigest()
    image_id = f"IPCIR_{messages_hash}_{image_idx}"
    return True, layout_info, messages_hash, image_id

def parse_json_layout(llm_output):
    """Parse JSON layout from LLM output

    Expected format: [{"label": "...", "desc": "...", "ref": "image|text"}]
    """
    try:
        # Remove markdown code blocks if present
        llm_output = re.sub(r'```json\s*', '', llm_output)
        llm_output = re.sub(r'```\s*', '', llm_output)

        # Try to find JSON array
        json_match = re.search(r'\[.*\]', llm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            layout_list = json.loads(json_str)

            # Validate format
            if isinstance(layout_list, list) and len(layout_list) > 0:
                validated_items = []
                for item in layout_list:
                    if not isinstance(item, dict):
                        print(f"  → Item is not a dict: {item}")
                        return None

                    # Check required fields
                    if 'label' not in item or 'desc' not in item:
                        print(f"  → Missing required fields (label/desc): {item}")
                        return None

                    # Ensure ref field exists (default to 'text' if missing)
                    if 'ref' not in item:
                        item['ref'] = 'text'

                    # Validate ref value
                    if item['ref'].lower() not in ['image', 'text']:
                        print(f"  → Invalid ref value (must be 'image' or 'text'): {item['ref']}")
                        item['ref'] = 'text'  # Default to text

                    # Normalize ref to lowercase
                    item['ref'] = item['ref'].lower()

                    validated_items.append(item)

                return validated_items

        print(f"  → No valid JSON array found in output")
        if len(llm_output) > 200:
            print(f"  → Output preview: {llm_output[:200]}...")
        else:
            print(f"  → Full output: {llm_output}")
        return None

    except json.JSONDecodeError as e:
        print(f"  → JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  → Unexpected error parsing layout: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_layout_random(args, classes, rule, scene, dataset_stastic):
    layout_info = get_random(args, classes, dataset_stastic, '')
    return layout_info

def get_advise(args, classes, rule, scene, layout, refer, model, tokenizer, prompt_system, examples):

    instruction_input = get_simple_inputs(classes, rule = rule, scene = scene, Initial_Layout = layout, refer = refer)
    basic_input = [get_chat_prompt(prompt_system, instruction_input, examples)]
    reasoning_output = apply_llm(args, basic_input, model, tokenizer)

    reasoning_info = reasoning_output[0].split('\n')
    advise_info = [i.strip().lower() for i in reasoning_info if i!='' and '##' in i]
    basic_layout_info = []
    has_scene = False
    final_scene = ''

    for advise_ in advise_info:
        instance = {}
        advise_list = [i for i in advise_.split('[##') if i!='']

        if len(advise_list) == 1 and 'scene' in advise_list[0]:
            key_ = advise_list[0].split(':')[0].strip().lower()
            value_ = advise_list[0].split(':')[-1].strip().lower().split('##]')[0].strip().lower()
            value_ = value_.split('.')[0]
            final_scene = value_
        else:
            for inst in advise_list:
                key_ = inst.split(':')[0].strip().lower()
                value_ = inst.split(':')[-1].strip().lower().split('##]')[0].strip().lower()
                if 'bbox' in key_:
                    try:
                        bbox = []
                        b_ = value_.split(',')
                        for _ in b_:
                            _ = float(_.strip().rstrip(";"))
                            bbox.append(_)
                        value_ = bbox
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area == 1.0:
                            has_scene = True
                            instance['is_scene'] = True
                        else:
                            instance['is_scene'] = False
                    except:
                        continue
                elif key_ == 'from':
                    if '0' in value_:
                        instance['from'] = 0
                    else:
                        instance['from'] = 1
                
                instance[key_] = value_
            if 'label' in instance and 'cate' in instance and 'desc' in instance and 'bbox' in instance and 'ref' in instance:
                basic_layout_info.append(instance)
            
    if has_scene == False and final_scene!='':
        basic_layout_info.append({'label':final_scene, 'cate':final_scene, 'is_scene':True, 'desc':final_scene,'bbox':[0.0,0.0,1.0,1.0],'ref':'text', 'size':5})
    return basic_layout_info


def get_simple_inputs(reference_name, rule = '', scene = [], Initial_Layout = [], refer = []):
    if len(refer) == 0:
        refer = ['text' for i in reference_name]
    input_list = []
    # if len(scene) > 0:
    object_list = ', '.join(reference_name)
    input_string = ''
    if len(scene) > 0:
        input_string = input_string + f'Scene: {scene[0]}'

    if len(reference_name) > 0:
        input_string = input_string + '\nObject: '
    idx = 0
    for obj in reference_name:
        input_string = input_string + f'Label : {obj}, reference: {refer[idx]}'
        if len(Initial_Layout) > idx and Initial_Layout[idx] is not None:
            input_string = input_string + f', Layout: {Initial_Layout[idx]}, '
        idx = idx + 1
    if rule!='':
        input_string = input_string + f'\nLayout Rule: {rule}'

    return input_string



