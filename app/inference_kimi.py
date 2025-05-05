from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import requests
from io import BytesIO
import torch
import os

# Set environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up quantization config for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_path = "moonshotai/Kimi-VL-A3B-Thinking"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    offload_folder="offload",
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load image from URL
response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
image = Image.open(BytesIO(response.content))
images = [image]  # Keep this for the processor

# Format messages with image in the user content - CORRECTED STRUCTURE
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful botanical assistant. You are very precise and you are very good at describing images."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image} for image in images
        ] + [{"type": "text", "text": "Describe this image in detail."}],
    }
]

# Process inputs with the correct format
text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)

# Use more conservative generation parameters
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)
