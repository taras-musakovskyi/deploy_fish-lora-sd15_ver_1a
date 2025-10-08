import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from peft import PeftModel
from cog import BasePredictor, Input, Path
import json
import os
import shutil

class Predictor(BasePredictor):
    def setup(self):
        """Load the model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        MODEL_NAME = "runwayml/stable-diffusion-v1-5"
        LORA_PATH = "models/fish_lora.safetensors"
        
        # Load base model
        print("Loading Stable Diffusion 1.5...")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        
        # Use EulerAncestralDiscreteScheduler like your notebook
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        print("Loading LoRA weights...")
        
        # Create PEFT structure
        temp_lora_dir = "/tmp/lora"
        os.makedirs(temp_lora_dir, exist_ok=True)
        shutil.copy(LORA_PATH, f"{temp_lora_dir}/adapter_model.safetensors")
        
        adapter_config = {
            "base_model_name_or_path": MODEL_NAME,
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "lora_dropout": 0.1,
        }
        with open(f"{temp_lora_dir}/adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        self.pipeline.unet = PeftModel.from_pretrained(
            self.pipeline.unet,
            temp_lora_dir,
            adapter_name="fish_lora"
        )
        
        self.pipeline = self.pipeline.to(self.device)
        print("Model ready")
    
    def predict(
        self,
        prompt: str = Input(
            description="Prompt for fish generation",
            default="a single fish in aquarium, ultra detailed scales and fins, sharp, high resolution"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="multiple fish, blur, deformed, mutated, metallic"
        ),
        num_inference_steps: int = Input(default=50, ge=20, le=100),
        guidance_scale: float = Input(default=6.5, ge=1.0, le=20.0),
        seed: int = Input(default=-1),
    ) -> Path:
        """Generate fish image"""
        
        generator = None if seed == -1 else torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
