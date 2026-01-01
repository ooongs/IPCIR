import argparse
import json
import os
import torch
import numpy as np
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


class LatentSaver:
    """Callback class to save intermediate latents at specific steps
    Solution from: https://github.com/huggingface/diffusers/discussions/6810
    """

    def __init__(self, pipe, target_steps, output_dirs, img_id, curr_count):
        self.pipe = pipe
        self.target_steps = sorted(target_steps)
        self.output_dirs = output_dirs
        self.img_id = img_id
        self.curr_count = curr_count
        self.current_step = 0
        self.saved_steps = []

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        """Callback function called at the end of each step"""
        self.current_step = step_index + 1

        # Check if we should save this step
        if self.current_step in self.target_steps:
            latents = callback_kwargs["latents"]

            # CRITICAL FIX: Check if VAE needs upcasting to FP32
            # This fixes the black image issue with SDXL VAE in FP16
            needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

            if needs_upcasting:
                self.pipe.upcast_vae()

            with torch.no_grad():
                # Convert latents to VAE's dtype
                latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

                # Decode using the scaling factor
                image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]

                # Convert to PIL image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = np.clip(image * 255, 0, 255).round().astype("uint8")
                pil_image = Image.fromarray(image[0])

                # Restore VAE dtype if it was upcasted
                if needs_upcasting:
                    self.pipe.vae.to(dtype=torch.float16)

            # Save image
            output_dir = self.output_dirs[self.current_step]
            save_path = os.path.join(output_dir, f'combined_{self.img_id}_{self.curr_count}.jpg')
            pil_image.save(save_path)
            self.saved_steps.append(self.current_step)

        return callback_kwargs


def generate_images_with_callback(pipe, prompt, target_steps, output_dirs, img_id, curr_count,
                                  max_steps=32, guidance_scale=7.5, seed=0):
    """Generate images and save intermediate steps using callback"""

    # Create callback
    callback = LatentSaver(pipe, target_steps, output_dirs, img_id, curr_count)

    # Generate with callback
    generator = torch.Generator("cuda").manual_seed(seed)

    output = pipe(
        prompt=prompt,
        num_inference_steps=max_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        generator=generator,
        callback_on_step_end=callback
    )

    return output.images[0], callback.saved_steps


def main():
    parser = argparse.ArgumentParser(description="Generate proxy images with intermediate steps using SDXL Base.")

    # Model path
    parser.add_argument('--model_path', default='stabilityai/stable-diffusion-xl-base-1.0', type=str,
                       help='Path to SDXL Base model')

    # Input/Output paths
    parser.add_argument('--json_file', default='./test1.json', type=str, help='Path to test1.json')
    parser.add_argument('--output_base_path', default='./output', type=str, help='Base output directory')

    # Generation parameters
    parser.add_argument('--img_per_prompt', default=1, type=int, help='Number of images to generate per prompt')
    parser.add_argument('--num_prompts', default=5, type=int, help='Number of prompts to use from multi_gpt-3.5_opt')
    parser.add_argument('--max_inference_steps', default=32, type=int, help='Maximum inference steps')
    parser.add_argument('--save_steps', nargs='+', type=int, default=[1, 4, 8, 16, 32],
                        help='Steps at which to save intermediate results (default: 1 4 8 16 32)')
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

    # Create output directories for each target step
    output_dirs = {}
    for step in args.save_steps:
        output_dir = os.path.join(args.output_base_path, f'proxy_images_sdxl_step{step}')
        combined_dir = os.path.join(output_dir, 'combined')
        os.makedirs(combined_dir, exist_ok=True)
        output_dirs[step] = combined_dir

    # Check existing generated images for each step
    exi_id = {}
    for step in args.save_steps:
        exi_id[step] = defaultdict(int)
        for img_name in os.listdir(output_dirs[step]):
            if img_name.endswith('.jpg'):
                try:
                    idx = img_name.split('_')[1]
                    exi_id[step][idx] += 1
                except:
                    continue

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
    print(f"Max inference steps: {args.max_inference_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Saving at steps: {args.save_steps}")
    print(f"Target per step: {args.num_prompts} prompts x {args.img_per_prompt} images = {args.num_prompts * args.img_per_prompt} images per ID")
    print("=" * 80)

    # Statistics tracking
    total_start_time = time.time()
    id_processing_times = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_images_generated = 0

    # Generate images
    pbar = tqdm(items_to_process, desc=f"GPU {args.idx}", position=args.idx)

    for item in pbar:
        img_id = str(item['reference_img_id'])

        # Check if all steps already have enough images for this ID
        all_steps_complete = all(
            exi_id[step][img_id] >= args.num_prompts * args.img_per_prompt
            for step in args.save_steps
        )

        if all_steps_complete:
            skipped_count += 1
            continue

        # Get prompts from multi_gpt-3.5_opt
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
                # Check if we need to generate for any step
                need_generation = False
                curr_counts = {}

                for step in args.save_steps:
                    curr_count = exi_id[step][img_id]
                    curr_counts[step] = curr_count
                    save_path = os.path.join(output_dirs[step], f'combined_{img_id}_{curr_count}.jpg')
                    if not os.path.exists(save_path):
                        need_generation = True
                        break

                if not need_generation:
                    # All steps already have this image
                    for step in args.save_steps:
                        exi_id[step][img_id] += 1
                    continue

                try:
                    # Use the first step's count for all (they should be in sync)
                    curr_count = curr_counts[args.save_steps[0]]

                    # Generate image with callback (saves all intermediate steps)
                    final_image, saved_steps = generate_images_with_callback(
                        pipe,
                        prompt,
                        target_steps=args.save_steps,
                        output_dirs=output_dirs,
                        img_id=img_id,
                        curr_count=curr_count,
                        max_steps=args.max_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=img_idx
                    )

                    # Update counts for all saved steps
                    for step in saved_steps:
                        exi_id[step][img_id] += 1

                    images_generated_for_id += len(saved_steps)
                    total_images_generated += len(saved_steps)

                except Exception as e:
                    pbar.write(f"Error generating image for item {img_id}, prompt {prompt_idx}: {e}")
                    import traceback
                    pbar.write(traceback.format_exc())
                    error_count += 1
                    continue

        # Record ID processing time
        if images_generated_for_id > 0:
            id_end_time = time.time()
            id_processing_times.append(id_end_time - id_start_time)
            processed_count += 1

            # Update progress bar with statistics
            avg_time_per_id = sum(id_processing_times) / len(id_processing_times)
            pbar.set_postfix({
                'Processed': processed_count,
                'Skipped': skipped_count,
                'Avg/ID': f'{avg_time_per_id:.2f}s',
                'Total_Imgs': total_images_generated
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
    print(f"Total Images Generated:    {total_images_generated}")
    print(f"Total Elapsed Time:        {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")

    if id_processing_times:
        avg_time_per_id = sum(id_processing_times) / len(id_processing_times)
        print(f"\nPer-ID Statistics:")
        print(f"  Average Time per ID:     {avg_time_per_id:.2f} seconds")

    # Print statistics per step
    print(f"\nImages generated per step:")
    for step in args.save_steps:
        total_imgs = sum(exi_id[step].values())
        print(f"  Step {step:2d}: {total_imgs} images")

    print("=" * 80)


if __name__ == "__main__":
    main()
