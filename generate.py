from PIL import Image
from glob import glob
from tqdm import tqdm
import os
import torch
import traceback
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Load processor and model
processor_flan = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model_flan = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.bfloat16)

# Move model to GPU if available
device_flan = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_flan.to(device_flan)

# Function to generate caption using flan model
def caption_flan(image, min_length=20):
    model_flan.to(device_flan)
    inputs = processor_flan(image, return_tensors="pt").to(device_flan, torch.bfloat16)
    generated_ids = model_flan.generate(
        **inputs,
        min_length=min_length,
        max_new_tokens=256,
        do_sample=False,
        num_beams=8,
        repetition_penalty=1.1
    )
    generated_text = processor_flan.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def unload_flan_model():
    global model_flan
    model_flan.to("cpu")
    torch.cuda.empty_cache()
    
    
print('generating captions')
try:
    imgs = glob(f"stick-man/*")
    for img_path in tqdm(imgs):
        ext = img_path.split('/')[-1].split('.')[-1]
        if ext in ['jpg','png','jpeg','txt','JPG','PNG','JPEG']:
            try:
                img = Image.open(img_path)
                caption = caption_flan(img)
                caption = "st1ckm4n, "+caption
                with open(img_path.replace(f'.{ext}','.txt'), 'w') as out:
                    out.write(caption)
            except:
                print("trace back:",str(traceback.format_exc()))
                print('failed to gen caption')
    unload_flan_model()
except:
    print("trace back:",str(traceback.format_exc()))
    print('failed to gen caption')
