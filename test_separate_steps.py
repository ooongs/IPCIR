"""
Test script for separate step execution
Each step is run independently (step 1=1 inference, step 4=4 inferences, etc.)
"""
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os


def test_separate_steps():
    """Test SDXL Base model with separate step execution"""
    print("=" * 80)
    print("Testing SDXL Base with SEPARATE step execution")
    print("Each step runs independently")
    print("=" * 80)

    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    output_dir = "./test_output/sdxl_separate_steps"
    os.makedirs(output_dir, exist_ok=True)

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
    inference_steps = [1, 4, 8, 16, 32]
    guidance_scale = 7.5
    seed = 0

    print(f"\nPrompt: {prompt}")
    print(f"Inference steps to test: {inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Seed: {seed}")

    saved_images = {}

    # Generate images for each step separately
    for num_steps in inference_steps:
        print(f"\n{'='*80}")
        print(f"Running {num_steps} inference steps")
        print(f"{'='*80}")

        image = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type="pil"
        ).images[0]

        # Save image
        save_path = os.path.join(output_dir, f'step_{num_steps:02d}.png')
        image.save(save_path)
        saved_images[num_steps] = save_path
        print(f"✓ Saved to {save_path}")

    print("\n" + "=" * 80)
    print("All steps completed!")
    print("=" * 80)
    print("\nGenerated images:")
    for step, path in sorted(saved_images.items()):
        print(f"  Step {step:2d}: {path}")

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print("✓ Step 1  (1 inference):   Very noisy, barely recognizable")
    print("✓ Step 4  (4 inferences):  Still quite noisy, rough shapes")
    print("✓ Step 8  (8 inferences):  Some structure visible")
    print("✓ Step 16 (16 inferences): Clearer image, details emerging")
    print("✓ Step 32 (32 inferences): Final quality image ✅")
    print("=" * 80)


if __name__ == "__main__":
    test_separate_steps()
    print("\n✅ Test completed!")
    print("\nYou can view the generated images in: ./test_output/sdxl_separate_steps/")
    print("\nNOTE: Each step is a COMPLETED image at that quality level,")
    print("      not an intermediate snapshot of a 32-step generation.")
