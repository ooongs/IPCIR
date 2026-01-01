"""
Test script with percentage-based step saving
This saves images at specific progress percentages (e.g., 25%, 50%, 75%, 100%)
regardless of total num_inference_steps
"""
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os


class LatentSaverPercentage:
    """Callback class to save intermediate latents at specific progress percentages"""

    def __init__(self, pipe, save_percentages, output_dir, total_steps):
        """
        Args:
            pipe: The diffusion pipeline
            save_percentages: List of percentages (0-1) to save, e.g., [0.25, 0.5, 0.75, 1.0]
            output_dir: Directory to save images
            total_steps: Total number of inference steps
        """
        self.pipe = pipe
        self.save_percentages = sorted(save_percentages)
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.current_step = 0
        self.saved_images = {}

        # Calculate actual step numbers from percentages
        self.target_steps = []
        for pct in self.save_percentages:
            step = int(round(pct * total_steps))
            step = max(1, min(step, total_steps))  # Clamp between 1 and total_steps
            self.target_steps.append(step)

        self.target_steps = sorted(set(self.target_steps))  # Remove duplicates
        print(f"Saving at steps: {self.target_steps} (from percentages: {[f'{p*100:.0f}%' for p in save_percentages]})")

        os.makedirs(output_dir, exist_ok=True)

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        """Callback function called at the end of each step"""
        self.current_step = step_index + 1
        progress = self.current_step / self.total_steps * 100

        print(f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%): timestep={timestep:.2f}")

        # Check if we should save this step
        if self.current_step in self.target_steps:
            latents = callback_kwargs["latents"]
            print(f"  → Saving step {self.current_step} ({progress:.0f}% progress, latent shape: {latents.shape})")

            # Decode latent to image
            try:
                # CRITICAL FIX: Check if VAE needs upcasting to FP32
                needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

                if needs_upcasting:
                    print(f"    Upcasting VAE to FP32")
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

                # Save image with progress percentage in filename
                save_path = os.path.join(self.output_dir, f'progress_{progress:03.0f}pct_step_{self.current_step:02d}.png')
                pil_image.save(save_path)
                self.saved_images[self.current_step] = {
                    'path': save_path,
                    'progress': progress
                }
                print(f"  ✓ Saved to {save_path}")

            except Exception as e:
                print(f"  ✗ Error saving step {self.current_step}: {e}")
                import traceback
                traceback.print_exc()

        return callback_kwargs


def test_sdxl_percentage(num_inference_steps=32):
    """Test SDXL Base model with percentage-based step saving"""
    print("=" * 80)
    print(f"Testing SDXL with {num_inference_steps} inference steps")
    print("Saving at progress percentages: 25%, 50%, 75%, 100%")
    print("=" * 80)

    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir = f"./test_output/sdxl_percentage_steps{num_inference_steps}"

    print(f"\nLoading SDXL Base model from {model_path}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    print("✓ Model loaded")

    prompt = "A cat holding a sign that says hello world"
    # Save at 25%, 50%, 75%, 100% progress
    save_percentages = [0.25, 0.5, 0.75, 1.0]
    guidance_scale = 7.5

    print(f"\nPrompt: {prompt}")
    print(f"Total steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")

    # Create callback
    callback = LatentSaverPercentage(pipe, save_percentages, output_dir, num_inference_steps)

    print("\nGenerating image with callback...")
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(0),
        callback_on_step_end=callback,
        output_type="pil"
    ).images[0]

    # Save final image
    final_path = os.path.join(output_dir, "final.png")
    image.save(final_path)
    print(f"\n✓ Final image saved to {final_path}")

    print("\n" + "=" * 80)
    print("Saved images:")
    print("=" * 80)
    for step, info in sorted(callback.saved_images.items()):
        print(f"  Step {step:2d} ({info['progress']:3.0f}%): {info['path']}")
    print(f"  Final     : {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test percentage-based step saving")
    parser.add_argument('--steps', type=int, default=32,
                       help='Number of inference steps (default: 32)')
    args = parser.parse_args()

    test_sdxl_percentage(num_inference_steps=args.steps)

    print(f"\n✅ Test completed with {args.steps} steps!")
    print(f"\nYou can view the generated images in: ./test_output/sdxl_percentage_steps{args.steps}/")
    print("\nTry running with different step counts to see the difference:")
    print("  python test_multistep_callback_percentage.py --steps 32")
    print("  python test_multistep_callback_percentage.py --steps 64")
