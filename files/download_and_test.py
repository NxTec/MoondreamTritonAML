import os
import sys
import shutil
from pathlib import Path


os.system(f"{sys.executable} -m pip install -q torch transformers tokenizers Pillow einops numpy")


from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "vikhyatk/moondream2"
revision = "2025-01-09"
output_dir = "./model_files"

try:
    os.makedirs(output_dir, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True
    )
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

py_files = list(Path(output_dir).glob("*.py"))
json_files = list(Path(output_dir).glob("*.json"))
other_files = [f for f in Path(output_dir).iterdir() if f.suffix not in ['.py', '.json']]



try:
    cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules" / output_dir.replace("./", "")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cache directory: {cache_dir}")
    
    copied_count = 0
    for py_file in py_files:
        shutil.copy2(py_file, cache_dir / py_file.name)
        copied_count += 1
    
    (cache_dir / "__init__.py").write_text("# Transformers cache module\n")
    
    for config_file in ["config.json", "generation_config.json"]:
        src = Path(output_dir) / config_file
        if src.exists():
            shutil.copy2(src, cache_dir / config_file)
    
    print(f"Copied {copied_count} Python files to cache")
    print(f"Created __init__.py in cache")
    
except Exception as e:
    print(f"Warning: Cache setup had issues: {e}")
    print("   Trying alternative loading method...")

print()

try:
    import torch
    from PIL import Image
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        torch.set_default_dtype(torch.float16)
    
    print("Loading model...")
    
    abs_output_dir = str(Path(output_dir).absolute())
    
    model = AutoModelForCausalLM.from_pretrained(
        abs_output_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model = model.to(device)
    model.eval()
    print(f"Model loaded (dtype: {next(model.parameters()).dtype})")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        abs_output_dir,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False
    )
    print("Tokenizer loaded")
    
    print("\nCreating test image (solid color 224x224)...")
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    print("Testing inference...")
    question = "What color is this image?"
    
    with torch.no_grad():
        result = model.query(test_image, question)
    
    if isinstance(result, dict):
        answer = result.get('answer', str(result))
    else:
        answer = str(result)
    

        
except Exception as e:
    print(f"  failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

