import argparse
import json
import os
import torch
import time
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import logging as diffusers_logging
from collections import defaultdict
from PIL import Image

# Disable diffusers progress bars
diffusers_logging.disable_progress_bar()


def load_sdxl_model(model_path, device='cuda'):
    """Load SDXL Base model"""
    import logging
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_images(pipe, prompt, num_images=1, num_inference_steps=1, guidance_scale=7.5, seed=0):
    """Generate images using SDXL pipeline"""
    images = []
    for i in range(num_images):
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="pil",
            generator=torch.Generator("cuda").manual_seed(seed + i)
        ).images[0]
        images.append(image)
    return images


def main():
    parser = argparse.ArgumentParser(description="Generate proxy images using SDXL Base with multiple inference steps.")

    # Model path
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-xl-base-1.0', type=str,
                       help='Path to SDXL Base model')

    # Input/Output paths
    parser.add_argument('--json_file', default='./test1.json', type=str, help='Path to test1.json')
    parser.add_argument('--output_base_path', default='./output', type=str, help='Base output directory')

    # Generation parameters
    parser.add_argument('--img_per_prompt', default=1, type=int, help='Number of images to generate per prompt')
    parser.add_argument('--num_prompts', default=5, type=int, help='Number of prompts to use from multi_gpt-3.5_opt')
    parser.add_argument('--inference_steps', nargs='+', type=int, default=[1, 4, 8, 16, 32],
                        help='List of inference steps to use (default: 1 4 8 16 32)')
    parser.add_argument('--guidance_scale', default=7.5, type=float, help='Guidance scale for SDXL')

    # Multi-GPU settings
    parser.add_argument('--idx', default=0, type=int, help='GPU index')
    parser.add_argument('--gpu_num', default=1, type=int, help='Total number of GPUs')

    args = parser.parse_args()

    run(args)


def run(args):
    # Set device
    device = torch.device('cuda')

    # Load SDXL Base model
    print(f"Loading SDXL Base model from {args.model_path}...")
    pipe = load_sdxl_model(args.model_path, device=device)
    print("Model loaded successfully!")

    # Load JSON data
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    # Split work across GPUs
    total_items = len(data)
    items_per_gpu = total_items // args.gpu_num

    if args.idx == args.gpu_num - 1:
        start_idx = args.idx * items_per_gpu
        end_idx = total_items
    else:
        start_idx = args.idx * items_per_gpu
        end_idx = (args.idx + 1) * items_per_gpu

    items_to_process = data[start_idx:end_idx]

    print("=" * 80)
    print(f"GPU {args.idx}: Processing items {start_idx} to {end_idx-1} ({len(items_to_process)} items)")
    print(f"Inference steps to generate: {args.inference_steps}")
    print(f"Target per step: {args.num_prompts} prompts x {args.img_per_prompt} images = {args.num_prompts * args.img_per_prompt} images per ID")
    print("=" * 80)

    # Process each inference step
    for num_inference_steps in args.inference_steps:
        print(f"\n{'='*80}")
        print(f"Processing Inference Step: {num_inference_steps}")
        print(f"{'='*80}")

        # Create output directory for this step
        output_dir = os.path.join(args.output_base_path, f'proxy_images_sdxl_step{num_inference_steps}')
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

        # Statistics tracking
        total_start_time = time.time()
        id_processing_times = []
        single_image_times = []
        processed_count = 0
        skipped_count = 0
        error_count = 0

        # Generate images
        pbar = tqdm(items_to_process, desc=f"GPU {args.idx} Step {num_inference_steps}", position=args.idx)

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
                            num_inference_steps=num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            seed=img_idx
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

        # Print statistics for this step
        print(f"\n{'='*80}")
        print(f"GPU {args.idx}: Step {num_inference_steps} Completed!")
        print(f"{'='*80}")
        print(f"Output Directory:          {output_dir}")
        print(f"Total Items Processed:     {processed_count}")
        print(f"Total Items Skipped:       {skipped_count}")
        print(f"Total Errors:              {error_count}")
        print(f"Total Images Generated:    {len(single_image_times)}")
        print(f"Total Elapsed Time:        {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")

        if id_processing_times:
            avg_time_per_id = sum(id_processing_times) / len(id_processing_times)
            print(f"\nPer-ID Statistics:")
            print(f"  Average Time per ID:     {avg_time_per_id:.2f} seconds")

        if single_image_times:
            avg_time_per_img = sum(single_image_times) / len(single_image_times)
            print(f"\nPer-Image Statistics:")
            print(f"  Average Time per Image:  {avg_time_per_img:.2f} seconds")
            print(f"  Images per Second:       {1/avg_time_per_img:.2f}")

        print(f"{'='*80}\n")

    print("\n" + "="*80)
    print("ALL INFERENCE STEPS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
