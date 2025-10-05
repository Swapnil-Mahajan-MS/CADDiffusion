import os, json, csv
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

# ---------- utils ----------
def find_images(root_dir):
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    root = Path(root_dir)
    imgs = []
    for ext in exts:
        imgs.extend(root.rglob(f"*{ext}"))
        imgs.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))

def load_existing_captions(out_jsonl):
    done = set()
    p = Path(out_jsonl)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done.add(rec["path"])
                except:
                    pass
    return done

def write_csv_from_jsonl(jsonl_file, csv_file):
    rows = []
    if not Path(jsonl_file).exists():
        return
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except:
                pass
    if rows:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path","caption"])
            w.writeheader()
            w.writerows(rows)

def load_image(path):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im

# ---------- main ----------
def main():
    # paths (change these)
    images_dir = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/Retrvial_Data/Images/joint"
    output_jsonl = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/Retrvial_Data/Images/llava_mistral_captions.jsonl"
    output_csv   = "/media/swapnil/3f73cc1a-8f9d-4c19-87af-99b3512ff5b2/Retrvial_Data/Images/llava_mistral_captions.csv"

    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"  # Mistral-7B backbone VLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading model:", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Use the correct model class for LLaVA v1.6
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device=="cuda" else None
    )
    if device == "cpu":
        model.to(device)

    # collect images & resume
    all_imgs = find_images(images_dir)
    print(f"Found {len(all_imgs)} images")
    done = load_existing_captions(output_jsonl)
    todo = [p for p in all_imgs if str(p) not in done]
    print(f"Already processed: {len(done)} | Remaining: {len(todo)}")
    if not todo:
        write_csv_from_jsonl(output_jsonl, output_csv)
        print("Nothing to do. CSV refreshed.")
        return

    # captioning loop - PROCESS ONE IMAGE AT A TIME
    PROMPT = "Describe this image concisely in one sentence for a CAD/technical audience."

    with open(output_jsonl, "a", encoding="utf-8") as out:
        for img_path in tqdm(todo, desc="Captioning"):
            try:
                image = load_image(img_path)
                
                # Simple prompt without chat template for LLaVA v1.6
                prompt = f"USER: <image>\n{PROMPT}\nASSISTANT:"

                # Process inputs - LLaVA v1.6 style
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )

                # Move to device
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(device)

                # Generate with proper settings
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        temperature=0.0,
                        use_cache=True
                    )

                # Decode and clean output
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                
                # Extract only the assistant's response
                if "ASSISTANT:" in generated_text:
                    caption = generated_text.split("ASSISTANT:")[-1].strip()
                elif "assistant" in generated_text.lower():
                    caption = generated_text.split("assistant")[-1].strip()
                else:
                    # Fallback - remove the original prompt
                    caption = generated_text.replace(prompt.replace("<image>", ""), "").strip()

                # Clean up any remaining artifacts
                caption = caption.replace("USER:", "").replace("ASSISTANT:", "").strip()
                
                if not caption:
                    caption = "No description generated"

                # Write to file
                out.write(json.dumps({"path": str(img_path), "caption": caption}, ensure_ascii=False) + "\n")
                out.flush()  # Ensure data is written immediately
                
                print(f"✓ {img_path.name}: {caption[:50]}...")
                
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
                # Write error record to continue processing
                out.write(json.dumps({"path": str(img_path), "caption": f"ERROR: {str(e)}"}, ensure_ascii=False) + "\n")
                out.flush()
                continue

    print("Done. Writing CSV…")
    write_csv_from_jsonl(output_jsonl, output_csv)
    print("Saved:\n  JSONL:", output_jsonl, "\n  CSV:", output_csv)

if __name__ == "__main__":
    main()
