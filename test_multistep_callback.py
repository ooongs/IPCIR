"""
Test script to verify intermediate step saving with callback
Based on flux.py
"""
import torch
from diffusers import FluxPipeline, AutoPipelineForText2Image
from PIL import Image
import os


class LatentSaverTest:
    """Test callback class to save intermediate latents at specific steps"""

    def __init__(self, pipe, target_steps, output_dir, model_type='flux'):
        self.pipe = pipe
        self.target_steps = sorted(target_steps)
        self.output_dir = output_dir
        self.model_type = model_type
        self.current_step = 0
        self.saved_images = {}
        os.makedirs(output_dir, exist_ok=True)

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        """Callback function called at the end of each step"""
        self.current_step = step_index + 1

        print(f"Step {self.current_step}/{len(callback_kwargs.get('timesteps', []))}: timestep={timestep}")

        # Check if we should save this step
        if self.current_step in self.target_steps:
            latents = callback_kwargs["latents"]
            print(f"  → Saving step {self.current_step} (latent shape: {latents.shape})")

            # Decode latent to image
            try:
                if self.model_type == 'flux':
                    # Flux: Different decoding process
                    latents = self.pipe._unpack_latents(latents, 512, 512, self.pipe.vae_scale_factor)
                    latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
                    with torch.no_grad():
                        image = self.pipe.vae.decode(latents).sample
                elif self.model_type == 'sdxl':
                    # SDXL: Scale latents before decoding
                    latents = latents / self.pipe.vae.config.scaling_factor
                    with torch.no_grad():
                        image = self.pipe.vae.decode(latents).sample

                # Convert to PIL image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = (image * 255).round().astype("uint8")
                pil_image = Image.fromarray(image[0])

                # Save image
                save_path = os.path.join(self.output_dir, f'step_{self.current_step:02d}.png')
                pil_image.save(save_path)
                self.saved_images[self.current_step] = save_path
                print(f"  ✓ Saved to {save_path}")

            except Exception as e:
                print(f"  ✗ Error saving step {self.current_step}: {e}")
                import traceback
                traceback.print_exc()

        return callback_kwargs


def test_flux():
    """Test Flux model with intermediate step saving"""
    print("=" * 80)
    print("Testing FLUX model with intermediate step saving")
    print("=" * 80)

    model_path = "/home/jinzhenxiong/pretrain/black-forest-labs/FLUX.1-schnell"
    output_dir = "./test_output/flux_multistep"

    print(f"\nLoading Flux model from {model_path}...")
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    print("✓ Model loaded")

    prompt = "A cat holding a sign that says hello world"
    target_steps = [1, 4, 8, 16, 32]
    max_steps = 32

    print(f"\nPrompt: {prompt}")
    print(f"Max steps: {max_steps}")
    print(f"Target steps to save: {target_steps}")

    # Create callback
    callback = LatentSaverTest(pipe, target_steps, output_dir, model_type='flux')

    print("\nGenerating image with callback...")
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=max_steps,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0),
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
    for step, path in sorted(callback.saved_images.items()):
        print(f"  Step {step:2d}: {path}")
    print(f"  Final  : {final_path}")
    print("=" * 80)


def test_sdxl():
    """Test SDXL model with intermediate step saving"""
    print("=" * 80)
    print("Testing SDXL model with intermediate step saving")
    print("=" * 80)

    model_path = "/home/jinzhenxiong/temp/stabilityai/sdxl-turbo"
    output_dir = "./test_output/sdxl_multistep"

    print(f"\nLoading SDXL model from {model_path}...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    print("✓ Model loaded")

    prompt = "A cat holding a sign that says hello world"
    target_steps = [1, 4, 8, 16, 32]
    max_steps = 32

    print(f"\nPrompt: {prompt}")
    print(f"Max steps: {max_steps}")
    print(f"Target steps to save: {target_steps}")

    # Create callback
    callback = LatentSaverTest(pipe, target_steps, output_dir, model_type='sdxl')

    print("\nGenerating image with callback...")
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=max_steps,
        generator=torch.Generator("cpu").manual_seed(0),
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
    for step, path in sorted(callback.saved_images.items()):
        print(f"  Step {step:2d}: {path}")
    print(f"  Final  : {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test intermediate step saving with callback")
    parser.add_argument('--model', default='flux', choices=['flux', 'sdxl', 'both'],
                       help='Which model to test (default: flux)')

    args = parser.parse_args()

    if args.model in ['flux', 'both']:
        test_flux()
        print("\n\n")

    if args.model in ['sdxl', 'both']:
        test_sdxl()
        print("\n\n")

    print("✅ Test completed!")
