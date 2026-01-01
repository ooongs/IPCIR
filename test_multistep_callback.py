"""
Test script to verify intermediate step saving with callback for SDXL Base model
Solution from: https://github.com/huggingface/diffusers/discussions/6810
"""
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os


class LatentSaverTest:
    """Test callback class to save intermediate latents at specific steps"""

    def __init__(self, pipe, target_steps, output_dir):
        self.pipe = pipe
        self.target_steps = sorted(target_steps)
        self.output_dir = output_dir
        self.current_step = 0
        self.saved_images = {}
        os.makedirs(output_dir, exist_ok=True)

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        """Callback function called at the end of each step"""
        self.current_step = step_index + 1

        print(f"Step {self.current_step}: timestep={timestep:.2f}")

        # Check if we should save this step
        if self.current_step in self.target_steps:
            latents = callback_kwargs["latents"]
            print(f"  → Saving step {self.current_step} (latent shape: {latents.shape})")

            # Decode latent to image
            try:
                # CRITICAL FIX: Check if VAE needs upcasting to FP32
                needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

                if needs_upcasting:
                    print(f"    Upcasting VAE to FP32 (fixes black image issue)")
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
                save_path = os.path.join(self.output_dir, f'step_{self.current_step:02d}.png')
                pil_image.save(save_path)
                self.saved_images[self.current_step] = save_path
                print(f"  ✓ Saved to {save_path}")

            except Exception as e:
                print(f"  ✗ Error saving step {self.current_step}: {e}")
                import traceback
                traceback.print_exc()

        return callback_kwargs


def test_sdxl():
    """Test SDXL Base model with intermediate step saving"""
    print("=" * 80)
    print("Testing SDXL Base model with intermediate step saving")
    print("=" * 80)

    model_path = "/home/jinzhenxiong/pretrain/stabilityai/stable-diffusion-xl-base-1.0"
    output_dir = "./test_output/sdxl_multistep"

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
    print(f"VAE dtype: {pipe.vae.dtype}")
    print(f"VAE force_upcast: {pipe.vae.config.force_upcast}")

    prompt = "A cat holding a sign that says hello world"
    target_steps = [1, 4]
    max_steps = 4
    guidance_scale = 7.5

    print(f"\nPrompt: {prompt}")
    print(f"Max steps: {max_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Target steps to save: {target_steps}")

    # Create callback
    callback = LatentSaverTest(pipe, target_steps, output_dir)

    print("\nGenerating image with callback...")
    image = pipe(
        prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=max_steps,
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
    for step, path in sorted(callback.saved_images.items()):
        print(f"  Step {step:2d}: {path}")
    print(f"  Final  : {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    test_sdxl()
    print("\n✅ Test completed!")
    print("\nYou can view the generated images in: ./test_output/sdxl_multistep/")
    print("Compare the images at different steps to see the denoising progression.")
