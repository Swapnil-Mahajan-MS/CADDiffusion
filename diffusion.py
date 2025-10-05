# CAD Training Script - Fixed CPU/GPU Transfer Issue
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import gc
from PIL import Image
import torchvision.transforms as transforms
import csv
from tqdm import tqdm

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

class Config:
    model_id = "stabilityai/stable-diffusion-2-1"
    resolution = 512
    train_batch_size = 1
    gradient_accumulation_steps = 64
    learning_rate = 1e-4
    max_train_steps = 50000
    mixed_precision = "fp16"
    
    lora_rank = 4
    lora_alpha = 8
    target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    data_dir = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/Retrvial_Data/Images/deepcad"
    csv_file = "/home/swapnil/Downloads/text2cad_v1.0_intermediate_clean.csv"
    output_dir = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/Retrvial_Data/Images/deepcad/cad_model_output"

def preprocess_caption(text):
    if not text:
        return "CAD model"
    text = text.replace("The CAD model consists of ", "")
    text = text.replace("This object is created by", "Created by")
    return f"CAD render: {text[:150]}"

class CADDataset:
    def __init__(self, data_dir, csv_file, tokenizer, size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        
        print(f"Loading dataset from: {csv_file}")
        print(f"Images directory: {data_dir}")
        
        self.data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            missing_count = 0
            empty_text_count = 0
            
            total_rows = sum(1 for line in open(csv_file, 'r', encoding='utf-8')) - 1
            f.seek(0)
            reader = csv.DictReader(f)
            
            for row in tqdm(reader, total=total_rows, desc="Loading dataset"):
                uid = row['uid']
                description = row['intermediate']
                image_path = os.path.join(data_dir, f"{uid}.png")
                
                if os.path.exists(image_path) and description.strip():
                    self.data.append({
                        'image_path': image_path,
                        'text': preprocess_caption(description)
                    })
                    count += 1
                    
                    if count >= 1000:  # Reduced for faster testing
                        break
                elif not os.path.exists(image_path):
                    missing_count += 1
                elif not description.strip():
                    empty_text_count += 1
        
        print(f"Successfully loaded {len(self.data)} valid image-text pairs")
        print(f"Missing images: {missing_count}")
        print(f"Empty descriptions: {empty_text_count}")
        
        if len(self.data) == 0:
            raise ValueError("No valid image-text pairs found! Check your paths.")
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=Image.LANCZOS),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            image = Image.open(item['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            image = torch.zeros(3, 512, 512)
        
        text_inputs = self.tokenizer(
            item['text'],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image.to(dtype=torch.float16),
            "input_ids": text_inputs.input_ids.squeeze()
        }

def main():
    config = Config()
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    cleanup_memory()
    os.makedirs(config.output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Load models
    print("Loading tokenizer...")
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    
    print("Loading text encoder...")
    from transformers import CLIPTextModel
    text_encoder = CLIPTextModel.from_pretrained(config.model_id, subfolder="text_encoder")
    text_encoder = text_encoder.to(accelerator.device, dtype=torch.float16)  # Keep on GPU
    text_encoder.requires_grad_(False)
    
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(config.model_id, subfolder="vae")
    vae = vae.to(accelerator.device, dtype=torch.float16)  # Keep on GPU
    vae.requires_grad_(False)
    
    print(f"After VAE load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    print("Loading UNet...")
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet")
    unet = unet.to(accelerator.device, dtype=torch.float16)
    
    print(f"After UNet load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Add LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.05,
    )
    
    unet = get_peft_model(unet, lora_config)
    unet.enable_gradient_checkpointing()
    
    print(f"After LoRA: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    print("LoRA trainable parameters:")
    unet.print_trainable_parameters()
    
    # Create dataset
    print("Creating dataset...")
    dataset = CADDataset(config.data_dir, config.csv_file, tokenizer, config.resolution)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_train_steps, eta_min=1e-6)
    
    # Prepare with accelerator
    unet, optimizer, dataloader, scheduler = accelerator.prepare(unet, optimizer, dataloader, scheduler)
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    
    # Training loop
    print("Starting training...")
    print(f"Total steps: {config.max_train_steps}")
    print(f"Effective batch size: {config.train_batch_size * config.gradient_accumulation_steps}")
    
    global_step = 0
    epoch = 0
    
    progress_bar = tqdm(total=config.max_train_steps, desc="Training")
    
    while global_step < config.max_train_steps:
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(epoch_bar):
            # print(f"Processing batch {batch_idx}")  # Debug print
            
            with accelerator.accumulate(unet):
                # Encode images to latents (no CPU/GPU transfers)
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings (no CPU/GPU transfers)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update global step
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                # Calculate additional metrics
                with torch.no_grad():
                    # Signal-to-noise ratio (higher = better)
                    signal_power = torch.mean(latents ** 2)
                    noise_power = torch.mean(noise ** 2)
                    snr = 10 * torch.log10(signal_power / noise_power)
                    
                    # Prediction accuracy (lower = better)
                    pred_error = torch.mean((noise_pred - noise) ** 2)
                    
                    # Learning progress (exponential moving average of loss)
                    if not hasattr(main, 'ema_loss'):
                        main.ema_loss = loss.item()
                    main.ema_loss = 0.99 * main.ema_loss + 0.01 * loss.item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ema_loss': f'{main.ema_loss:.4f}',
                    'pred_err': f'{pred_error.item():.4f}',
                    'snr': f'{snr.item():.1f}dB',
                    'VRAM': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
                
                if global_step % 50 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    tqdm.write(f"Step {global_step:5d} | Loss: {loss.item():.4f} | "
                              f"LR: {current_lr:.2e} | VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                
                # Save checkpoint
                if global_step % 2000 == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unet_lora = accelerator.unwrap_model(unet)
                    lora_state_dict = {k: v for k, v in unet_lora.state_dict().items() if "lora" in k}
                    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pth"))
                    
                    import json
                    config_dict = {
                        "model_id": config.model_id,
                        "resolution": config.resolution,
                        "lora_rank": config.lora_rank,
                        "lora_alpha": config.lora_alpha,
                        "target_modules": config.target_modules,
                        "global_step": global_step
                    }
                    with open(os.path.join(save_path, "config.json"), "w") as f:
                        json.dump(config_dict, f, indent=2)
                    
                    tqdm.write(f"Checkpoint saved at step {global_step}")
                
                if global_step >= config.max_train_steps:
                    break
        
        epoch += 1
        epoch_bar.close()
        
        if global_step >= config.max_train_steps:
            break
    
    progress_bar.close()
    
    # Final save
    print("Training completed! Saving final model...")
    final_path = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    
    unet_lora = accelerator.unwrap_model(unet)
    lora_state_dict = {k: v for k, v in unet_lora.state_dict().items() if "lora" in k}
    torch.save(lora_state_dict, os.path.join(final_path, "lora_weights.pth"))
    
    print(f"Final model saved to: {final_path}")
    print(f"Training completed after {global_step} steps!")

if __name__ == "__main__":
    main()
