import argparse
import asyncio
import json
import logging
import re
from tqdm import tqdm
from enum_definitions import *
from data_definitions import *
from typing import Iterator
import yaml
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import clip

from torch.utils.data import Dataset
from torch.nn import Module, Parameter
import time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
from src.data_utils import collate_fn, PROJECT_ROOT, targetpad_transform
from src.datasets_fuck import FashionIQDataset, CIRRDatasetV2, CIRCODatasetV2
from layout_utils.llm_utils import *
from layout_utils.utils import *



async def main():
    parser = argparse.ArgumentParser(description="Generate a question-answer instruction tuning dataset.")

    # Mandatory parameter
    parser.add_argument("--sources",default='coco',type=str)
    
    ###### File Path in Generation
    parser.add_argument('--output_path',default='',type=str)
    parser.add_argument('--llm_path',default='',type=str)
    parser.add_argument('--prompt_config',default='./prompt/prompt_layout_v2.yaml', type=str)
    ###### Layout Generation
    parser.add_argument('--model_type',default='qwen',type=str)
    parser.add_argument("--batch_size",default=1,type=int)
    ###### Debug
    parser.add_argument('--visualize',action='store_true',help = 'Visualize the generated layouts')
    parser.add_argument('--visualize_path',default='./visual_layout/',type=str)
    
    ### Dataset Parameters
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument('--mode',default='circo_test',type=str)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=False, default='/home/llq/WorkSpace/code/Course/VL/dataset/CIRCO')

    args = parser.parse_args()

    # Placeholder for the dataset generation logic.
    print("Generating Proxy Layout with the following parameters:")
    print("Dataset_mode: ", args.mode)
    print(f'The Generated layout with be saved in {args.output_path}')
    print(f'Generate Layout with {args.model_type}')

    clip_model, clip_preprocess = clip.load('ViT-L/14', device='cpu', jit=False, download_root=os.path.join(PROJECT_ROOT, 'weights'))

    if args.preprocess_type == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    elif args.preprocess_type == 'clip':
        print('CLIP preprocess pipeline is used')
        preprocess = clip_preprocess
    else:
        raise ValueError("Preprocess type not supported")


    if args.mode == 'fashioniq_shirt':
        # for dress_type in ['shirt', 'dress', 'toptee']:
        relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['shirt'], 'relative', preprocess)
        await run(relative_val_dataset, preprocess, args)
    if args.mode == 'fashioniq_dress':
        relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['dress'], 'relative', preprocess)
        await run(relative_val_dataset, preprocess, args)
    if args.mode == 'fashioniq_toptee':
        relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['toptee'], 'relative', preprocess)
        await run(relative_val_dataset, preprocess, args)

    elif args.mode == 'cirr':
        relative_val_dataset = CIRRDatasetV2(args.dataset_path, 'val', 'relative', preprocess)
        await run(relative_val_dataset, preprocess, args)
    elif args.mode == 'cirr_test':
        relative_test_dataset = CIRRDatasetV2(args.dataset_path, 'test', 'relative', preprocess)
        await run(relative_test_dataset, preprocess, args)
    elif args.mode == 'circo':
        relative_val_dataset = CIRCODatasetV2(args.dataset_path, 'val', 'relative', preprocess)
        await run(relative_val_dataset, preprocess, args)
    elif args.mode == 'circo_test':
        relative_test_dataset = CIRCODatasetV2(args.dataset_path, 'test', 'relative', preprocess)
        await run(relative_test_dataset, preprocess, args)
    else:
        await run(None, None, args)

async def run(dataset, preprocess, args):

    dataset_stastic = {}
    device = torch.device(f'cuda')
    tokenizer, model = load_model_llm(args, device = device)

    prompt_configs = load_config(args)
    prompt_system = prompt_configs['layout']['system_prompt']
    examples = prompt_configs['layout']['examples']

    instance_count = 0
    layout_instance = {}

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    relative_val_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=10,
                                pin_memory=False, collate_fn=collate_fn, shuffle=False)

    # Compute the features
    idx = 0
    inst_id = 0
    for batch in tqdm(relative_val_loader):

        if isinstance(batch, dict):
            reference_names = batch['reference_name']
            relative_captions = batch['relative_caption']
            concept = batch['multi_opt'][0]
            target = batch['multi_gpt_opt'][0]
            bsz = len(concept)
        else:
            reference_names = None
            relative_captions = None
            concept = None
            target = None
            bsz = 1

        async def run_one(bs):
            # Keep same assertions as before to preserve behavior
            assert reference_names is not None, "For layout generation, reference names must be provided in the dataset."
            assert relative_captions is not None, "For layout generation, relative captions must be provided in the dataset."
            assert concept is not None, "For layout generation, concept must be provided in the dataset."
            assert target is not None, "For layout generation, target must be provided in the dataset."
            while True: # Generate util has a answer
                if isinstance(concept[bs], tuple):
                    classes = [concept[bs][0]]
                else:
                    classes = [concept[bs]]

                refer = ['image' for i in classes]
                scene = []
                rule = relative_captions[bs]
                if len(target) > 0:
                    tar_ = target[bs]
                    rule = f'generating a image of {tar_}'
                layout = []
                flag, layout_info, messages_hash, question_id = await llm_layout_async(args, classes, rule, scene, layout, refer, model, tokenizer, idx, dataset_stastic, prompt_system, examples)

                if flag == True:
                    break
            
            return layout_info, question_id
        
        tasks = [run_one(bs) for bs in range(bsz)]
        results = await asyncio.gather(*tasks)

        for bs, (layout_info, question_id) in enumerate(results):
            if 'pair_id' in batch.keys():
                question_id = batch['pair_id'].tolist()[bs]
            else:
                assert reference_names is not None, "For layout generation, reference names must be provided in the dataset."
                question_id = reference_names[bs]

            idx = idx + 1
            if layout_info == None:
                continue
            inst_id = inst_id + 1
            
            layout_instance[question_id] = {
                "layout": layout_info,
                "name":question_id,
            }

    json_output_path = os.path.join(args.output_path, f'output_proxy_layout_0.json')
    with open(json_output_path,'w') as f:
        json.dump(layout_instance, f, indent=4)
    
if __name__ == "__main__":
    asyncio.run(main())