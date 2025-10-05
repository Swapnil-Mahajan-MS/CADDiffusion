# CAD Model Test Script
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import os
from datetime import datetime

class CADGenerator:
    def __init__(self, lora_weights_path, output_dir="generated_images"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading base model...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(self.device)
        
        print("Loading LoRA weights...")
        self.load_lora_weights(lora_weights_path)
        
        print("Model ready!")
    
    def load_lora_weights(self, weights_path):
        # Load LoRA weights
        lora_weights = torch.load(weights_path, map_location=self.device)
        
        # Apply LoRA to UNet
        self.pipe.unet.load_attn_procs(lora_weights)
    
    def generate_image(self, prompt, filename=None, num_inference_steps=20, guidance_scale=7.5):
        print(f"Generating: {prompt}")
        
        # Add CAD-specific prompt prefix if not present
        if not prompt.startswith("CAD render"):
            prompt = f"CAD render: {prompt}"
        
        # Generate image
        with torch.no_grad():
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt="blurry, noisy, low quality, distorted, artifacts"
            ).images[0]
        
        # Save image
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cad_generated_{timestamp}.png"
        
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        
        filepath = os.path.join(self.output_dir, filename)
        image.save(filepath)
        print(f"Image saved: {filepath}")
        
        return image, filepath

def main():
    # Configuration
    LORA_WEIGHTS_PATH = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/Retrvial_Data/Images/deepcad/cad_model_output/checkpoint-8000/lora_weights.pth"
    OUTPUT_DIR = "/home/swapnil/Desktop/Placement26/ADAS/test_generated_images"
    
    # Initialize generator
    generator = CADGenerator(LORA_WEIGHTS_PATH, OUTPUT_DIR)
    
    # Test prompts
    test_prompts = [
        "The CAD model consists of a hollow cylindrical structure with a flat base and tapered top. This object is created by first setting up a coordinate system, then sketching two complex loops on the X-Y plane. The sketch is then extruded along the normal direction to form a solid body. The resulting part has a height of approximately 0.75 units."
    ]
    
    print(f"\nGenerating {len(test_prompts)} test images...")
    
    for i, prompt in enumerate(test_prompts):
        filename = f"test_{i+1:02d}_{prompt[:20].replace(' ', '_')}.png"
        generator.generate_image(prompt, filename)
    
    print(f"\nAll images saved to: {OUTPUT_DIR}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("Enter text prompts to generate CAD images")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        user_prompt = input("\nEnter CAD description: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_prompt:
            try:
                generator.generate_image(user_prompt)
            except Exception as e:
                print(f"Error generating image: {e}")
        else:
            print("Please enter a valid prompt")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()
