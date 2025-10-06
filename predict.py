import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from cog import BasePredictor, Input, Path
import json
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load base model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        
        # Faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Setup PEFT structure
        lora_path = "models"
        adapter_config_path = os.path.join(lora_path, "adapter_config.json")
        
        if not os.path.exists(adapter_config_path):
            print("Creating adapter_config.json...")
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
        
        # Rename safetensors if needed
        lora_file = os.path.join(lora_path, "fish_lora.safetensors")
        adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
        if os.path.exists(lora_file) and not os.path.exists(adapter_file):
            os.rename(lora_file, adapter_file)
        
        # Load PEFT LoRA
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            lora_path
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        print("Model ready")
    
    def predict(
        self,
        prompt: str = Input(description="Prompt", default="a beautiful fish in aquarium"),
        num_inference_steps: int = Input(default=30, ge=20, le=50),
        guidance_scale: float = Input(default=7.5, ge=1.0, le=15.0),
        seed: int = Input(default=-1),
    ) -> Path:
        """Generate image"""
        
        generator = None if seed == -1 else torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)