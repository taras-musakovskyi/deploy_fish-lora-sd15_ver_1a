"""
Fish LoRA Model for Replicate
Generates aquarium fish images using trained Stable Diffusion LoRA
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from PIL import Image
import os
from typing import List
from cog import BasePredictor, Input, Path
import json

class Predictor(BasePredictor):
    """Predictor class for Cog"""
    
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Stable Diffusion 1.5 base model...")
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load base SD 1.5 pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        
        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        print("Loading LoRA weights...")
        
        # Path to LoRA model
        lora_path = "models"
        
        # Check if LoRA directory exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA model directory not found at {lora_path}")
        
        # Check if it's a directory (PEFT format) or needs config creation
        if os.path.isdir(lora_path):
            # Check for adapter files
            adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")
            
            if not os.path.exists(adapter_model_path):
                # Try with the full filename
                adapter_model_path = os.path.join(lora_path, "fish_lora.safetensors")
                
                if os.path.exists(adapter_model_path):
                    # Create PEFT structure
                    print("Converting single file to PEFT format...")
                    os.rename(adapter_model_path, os.path.join(lora_path, "adapter_model.safetensors"))
                    
                    # Create adapter config
                    adapter_config = {
                        "base_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                        "peft_type": "LORA",
                        "r": 16,
                        "lora_alpha": 32,
                        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                        "lora_dropout": 0.1,
                        "bias": "none",
                        "task_type": "FEATURE_EXTRACTION"
                    }
                    
                    with open(adapter_config_path, "w") as f:
                        json.dump(adapter_config, f, indent=2)
                    
                    print("✓ PEFT structure created")
        
        # Load LoRA into UNet
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            lora_path,
            adapter_name="fish_lora"
        )
        
        print("✓ LoRA loaded successfully")
        
        # Move to GPU
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            self.pipe.enable_xformers_memory_efficient_attention()
        
        print("✓ Model ready for inference")
        
        # Default negative prompt (always applied)
        self.default_negative_prompt = (
            "metallic, shiny, reflective, mirror-like, overexposed, "
            "blurry, distorted, deformed, ugly, bad anatomy, "
            "low quality, worst quality"
        )
    
    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for fish generation",
            default="a beautiful fish in aquarium"
        ),
        num_images: int = Input(
            description="Number of images to generate",
            default=1,
            ge=1,
            le=9
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (more = higher quality but slower)",
            default=30,
            ge=20,
            le=50
        ),
        guidance_scale: float = Input(
            description="Guidance scale (higher = more prompt following)",
            default=6.5,
            ge=1.0,
            le=15.0
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        
        print(f"Generating {num_images} image(s) with prompt: {prompt}")
        
        # Set seed if specified
        if seed == -1:
            seed = None
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")
        
        # Generate images
        output = self.pipe(
            prompt=[prompt] * num_images,
            negative_prompt=[self.default_negative_prompt] * num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Save outputs
        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"/tmp/output_{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))
            print(f"✓ Saved image {i+1}/{num_images}")
        
        return output_paths
