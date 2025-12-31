import argparse
import json
import os
import torch
import time
from tqdm import tqdm
from diffusers import AutoPipelineForText2Image, FluxPipeline
from diffusers.utils import logging as diffusers_logging
from collections import defaultdict
from PIL import Image

# Disable diffusers progress bars
diffusers_logging.disable_progress_bar()


def load_sdxl_model(model_path, device='cuda'):
    """Load Flux model"""
    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def load_flux_model(model_path, device='cuda'):
    """Load Flux model"""
    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def generate_images(pipe, prompt, num_images=1, num_inference_steps=1, guidance_scale=0.0):
    """Generate images using model pipeline"""
    images = []
    for i in range(num_images):
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pil",
            generator=torch.Generator("cpu").manual_seed(i)
        ).images[0]
        images.append(image)
    return images


def main():
    parser = argparse.ArgumentParser(description="Generate proxy images using SDXL.")
    parser.add_argument('--model_type', default='sdxl', type=str, choices=['sdxl', 'flux'])
    # Model paths
    parser.add_argument('--sdxl_path', default='/home/jinzhenxiong/temp/stabilityai/sdxl-turbo', type=str)
    parser.add_argument('--flux_path', default='/home/jinzhenxiong/pretrain/black-forest-labs/FLUX.1-schnell', type=str)

    # Input/Output paths
    parser.add_argument('--json_file', default='./test1.json', type=str, help='Path to test1.json')
    parser.add_argument('--output_path', default='./output/proxy_images_sdxl', type=str)

    # Generation parameters
    parser.add_argument('--img_per_prompt', default=1, type=int, help='Number of images to generate per prompt')
    parser.add_argument('--num_prompts', default=5, type=int, help='Number of prompts to use from multi_gpt-3.5_opt')
    parser.add_argument('--num_inference_steps', default=1, type=int)
    parser.add_argument('--guidance_scale', default=0.0, type=float)

    # Multi-GPU settings
    parser.add_argument('--idx', default=0, type=int, help='GPU index')
    parser.add_argument('--gpu_num', default=1, type=int, help='Total number of GPUs')

    args = parser.parse_args()

    run(args)


def run(args):
    # Set device
    device = torch.device('cuda')

    # Load SDXL model
    print(f"Loading {args.model_type} model from {args.sdxl_path}...")
    if args.model_type == 'sdxl':
        pipe = load_sdxl_model(args.sdxl_path, device=device)
    elif args.model_type == 'flux':
        pipe = load_flux_model(args.flux_path, device=device)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    print("Model loaded successfully!")

    # Load JSON data
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # Create output directory
    output_dir = args.output_path
    combined_dir = os.path.join(output_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)

    # Check existing generated images
    exi_id = defaultdict(int)
    for img_name in os.listdir(combined_dir):
        if img_name.endswith('.jpg'):
            try:
                idx = img_name.split('_')[1]
                exi_id[idx] += 1
            except:
                continue

    # Split work across GPUs
    total_items = len(data)
    items_per_gpu = total_items // args.gpu_num

    if args.idx == args.gpu_num - 1:
        # Last GPU handles remaining items
        start_idx = args.idx * items_per_gpu
        end_idx = total_items
    else:
        start_idx = args.idx * items_per_gpu
        end_idx = (args.idx + 1) * items_per_gpu

    items_to_process = data[start_idx:end_idx]
    print(f"GPU {args.idx}: Processing items {start_idx} to {end_idx-1} ({len(items_to_process)} items)")
    print(f"GPU {args.idx}: Target - {args.num_prompts} prompts x {args.img_per_prompt} images = {args.num_prompts * args.img_per_prompt} images per ID")
    print("-" * 80)

    # Statistics tracking
    total_start_time = time.time()
    id_processing_times = []
    single_image_times = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Generate images
    pbar = tqdm(items_to_process, desc=f"GPU {args.idx}", position=args.idx)

    for item in pbar:
        img_id = str(item['reference_img_id'])

        # Check if already generated enough images
        if exi_id[img_id] >= args.num_prompts * args.img_per_prompt:
            skipped_count += 1
            continue

        # Get prompts from multi_gpt-3.5_opt (first num_prompts)
        prompts = item.get('multi_gpt-3.5_opt', [])[:args.num_prompts]

        if not prompts:
            pbar.write(f"Warning: No prompts found for item {img_id}")
            error_count += 1
            continue

        # Track time for this ID
        id_start_time = time.time()
        images_generated_for_id = 0

        # Generate images for each prompt
        for prompt_idx, prompt in enumerate(prompts):
            for img_idx in range(args.img_per_prompt):
                # Check if this specific image already exists
                curr_count = exi_id[img_id]
                save_name = os.path.join(
                    combined_dir,
                    f'combined_{img_id}_{curr_count}.jpg'
                )

                if os.path.exists(save_name):
                    exi_id[img_id] += 1
                    continue

                try:
                    # Generate image and track time
                    img_start_time = time.time()
                    images = generate_images(
                        pipe,
                        prompt,
                        num_images=1,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale
                    )
                    img_end_time = time.time()

                    # Save image
                    images[0].save(save_name)
                    exi_id[img_id] += 1
                    images_generated_for_id += 1

                    # Record time
                    single_image_times.append(img_end_time - img_start_time)

                except Exception as e:
                    pbar.write(f"Error generating image for item {img_id}, prompt {prompt_idx}: {e}")
                    error_count += 1
                    continue

        # Record ID processing time
        if images_generated_for_id > 0:
            id_end_time = time.time()
            id_processing_times.append(id_end_time - id_start_time)
            processed_count += 1

            # Update progress bar with statistics
            avg_time_per_id = sum(id_processing_times) / len(id_processing_times)
            avg_time_per_img = sum(single_image_times) / len(single_image_times) if single_image_times else 0
            pbar.set_postfix({
                'Processed': processed_count,
                'Skipped': skipped_count,
                'Avg/ID': f'{avg_time_per_id:.2f}s',
                'Avg/Img': f'{avg_time_per_img:.2f}s'
            })

    pbar.close()
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    # Print final statistics
    print("\n" + "=" * 80)
    print(f"GPU {args.idx}: Generation Completed!")
    print("=" * 80)
    print(f"Total Items Processed:     {processed_count}")
    print(f"Total Items Skipped:       {skipped_count}")
    print(f"Total Errors:              {error_count}")
    print(f"Total Images Generated:    {len(single_image_times)}")
    print(f"Total Elapsed Time:        {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")

    if id_processing_times:
        avg_time_per_id = sum(id_processing_times) / len(id_processing_times)
        min_time_per_id = min(id_processing_times)
        max_time_per_id = max(id_processing_times)
        print(f"\nPer-ID Statistics ({args.num_prompts * args.img_per_prompt} images per ID):")
        print(f"  Average Time per ID:     {avg_time_per_id:.2f} seconds")
        print(f"  Min Time per ID:         {min_time_per_id:.2f} seconds")
        print(f"  Max Time per ID:         {max_time_per_id:.2f} seconds")

    if single_image_times:
        avg_time_per_img = sum(single_image_times) / len(single_image_times)
        min_time_per_img = min(single_image_times)
        max_time_per_img = max(single_image_times)
        print(f"\nPer-Image Statistics:")
        print(f"  Average Time per Image:  {avg_time_per_img:.2f} seconds")
        print(f"  Min Time per Image:      {min_time_per_img:.2f} seconds")
        print(f"  Max Time per Image:      {max_time_per_img:.2f} seconds")
        print(f"  Images per Second:       {1/avg_time_per_img:.2f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
