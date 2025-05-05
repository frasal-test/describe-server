from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import requests
from io import BytesIO
import torch
import os
from typing import List, Dict, Union, Optional
from dotenv import load_dotenv

class KimiVLInference:
    """
    A class for performing inference with the Kimi Vision-Language model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Kimi VL model.
        
        Args:
            model_path: Optional model path. If not provided, will use the MODEL_PATH from .env
        """
        # Load environment variables
        load_dotenv()
        
        # Set environment variable to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("Loading model..")
        # Get model path from environment variable if not provided
        self.model_path = model_path or os.getenv("KIMI_MODEL_PATH", "moonshotai/Kimi-VL-A3B-Instruct")
        
        # Set up quantization config for memory efficiency (8bit)
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        # Set up quantization config for memory efficiency (4bit)
        # self.quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # )
        
        # Load model and processor
        self.model = self._load_model()
        self.processor = self._load_processor()
    
    def _load_model(self):
        """Load the model with appropriate configuration."""
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True  # Add this parameter to the config
            ),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            offload_folder="offload",
        )
    
    def _load_processor(self):
        """Load the processor for the model."""
        return AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """
        Load an image from a URL.
        
        Args:
            url: URL of the image
            
        Returns:
            PIL Image object
        """
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    
    def format_messages(self, 
                        images: List[Image.Image], 
                        prompt: str, 
                        system_prompt: Optional[str] = None) -> List[Dict]:
        """
        Format messages with images and text for the model.
        
        Args:
            images: List of PIL Image objects
            prompt: Text prompt to accompany the images
            system_prompt: Optional system prompt to set the behavior of the model
            
        Returns:
            Formatted messages list
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Add user message with images and prompt
        image_content = [{"type": "image", "image": image} for image in images]
        text_content = [{"type": "text", "text": prompt}]
        
        messages.append({
            "role": "user",
            "content": image_content + text_content,
        })
        
        return messages
    
    def generate_response(self, 
                          images: List[Image.Image], 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          max_new_tokens: int = 4000,
                          temperature: float = 0.7,
                          top_p: float = 0.9) -> str:
        """
        Generate a response from the model based on images and text.
        
        Args:
            images: List of PIL Image objects
            prompt: Text prompt to accompany the images
            system_prompt: Optional system prompt to set the behavior of the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text response
        """
        # Format messages
        messages = self.format_messages(images, prompt, system_prompt)
        
        # Process inputs
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        
        # Check if images list is empty - if so, don't pass images to processor
        if not images:
            # For text-only processing
            inputs = self.processor(
                text=text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
        else:
            # For image + text processing
            inputs = self.processor(
                images=images, 
                text=text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.model.device)
        
        # Generate response
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
        # Decode the response
        response = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response