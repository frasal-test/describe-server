from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from typing import Optional, List
import logging
import os
import time
from dotenv import load_dotenv
from PIL import Image
import io
from app.kimi_vl_lib import KimiVLInference
from app.api_logger import APILogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("API_KEY", "default_api_key")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# Initialize FastAPI app
app = FastAPI(title="Kimi VL API", description="API for Kimi Vision-Language Model")

# Initialize model
model = KimiVLInference()

# Create a lock for GPU access
gpu_lock = asyncio.Lock()

# Request and response models
class TextGenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerateResponse(BaseModel):
    output: str

# API key verification dependency
def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# Middleware to log request and response
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    endpoint = request.url.path
    
    # Skip logging for health check endpoint
    if endpoint == "/health":
        return await call_next(request)
    
    # Start timer
    start_time = time.time()
    
    # Process the request
    try:
        response = await call_next(request)
        execution_time = time.time() - start_time
        
        # Log successful response
        APILogger.log_response(client_ip, endpoint, "Response sent successfully", execution_time)
        
        return response
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Log error
        APILogger.log_error(client_ip, endpoint, str(e))
        raise

@app.post("/generate/text", response_model=GenerateResponse)
async def generate_text(data: TextGenerateRequest, request: Request, _: bool = Depends(verify_api_key)):
    """Generate text response without images"""
    client_ip = request.client.host
    endpoint = request.url.path
    
    # Log request
    APILogger.log_request(client_ip, endpoint, data.dict())
    
    start_time = time.time()
    
    async with gpu_lock:
        logger.info(f"Received text request with prompt: {data.prompt[:100]}...")
        
        # Use empty list for images since this endpoint is for text-only
        max_tokens = data.max_tokens or MAX_TOKENS
        
        output = model.generate_response(
            images=[],  # No images for this endpoint
            prompt=data.prompt,
            system_prompt=data.system_prompt,
            max_new_tokens=max_tokens,
            temperature=data.temperature,
            top_p=data.top_p
        )
        
        execution_time = time.time() - start_time
        
        # Log response
        APILogger.log_response(client_ip, endpoint, output[:100] + "...", execution_time)
        
        logger.info("Response generated successfully")
        return {"output": output}

@app.post("/generate/image", response_model=GenerateResponse)
async def generate_image_response(
    request: Request,
    prompt: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(0.7),
    top_p: Optional[float] = Form(0.9),
    images: List[UploadFile] = File(...),
    _: bool = Depends(verify_api_key)
):
    """Generate response based on images and text prompt"""
    client_ip = request.client.host
    endpoint = request.url.path
    
    # Log request
    request_data = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "images": [img.filename for img in images]
    }
    APILogger.log_request(client_ip, endpoint, request_data)
    
    start_time = time.time()
    
    async with gpu_lock:
        logger.info(f"Received image request with prompt: {prompt[:100]}...")
        
        # Process uploaded images
        pil_images = []
        for img in images:
            content = await img.read()
            pil_image = Image.open(io.BytesIO(content))
            pil_images.append(pil_image)
        
        logger.info(f"Processing {len(pil_images)} images")
        
        # Set max tokens
        max_tokens_value = max_tokens or MAX_TOKENS
        
        # Generate response
        output = model.generate_response(
            images=pil_images,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens_value,
            temperature=temperature,
            top_p=top_p
        )
        
        execution_time = time.time() - start_time
        
        # Log response
        APILogger.log_response(client_ip, endpoint, output[:100] + "...", execution_time)
        
        logger.info("Response generated successfully")
        return {"output": output}

@app.post("/generate/image_url", response_model=GenerateResponse)
async def generate_from_url(
    request: Request,
    prompt: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(0.7),
    top_p: Optional[float] = Form(0.9),
    image_url: str = Form(...),
    _: bool = Depends(verify_api_key)
):
    """Generate response based on image URL and text prompt"""
    client_ip = request.client.host
    endpoint = request.url.path
    
    # Log request
    request_data = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "image_url": image_url
    }
    APILogger.log_request(client_ip, endpoint, request_data)
    
    start_time = time.time()
    
    async with gpu_lock:
        logger.info(f"Received image URL request with prompt: {prompt[:100]}...")
        
        try:
            # Load image from URL
            image = model.load_image_from_url(image_url)
            
            # Set max tokens
            max_tokens_value = max_tokens or MAX_TOKENS
            
            # Generate response
            output = model.generate_response(
                images=[image],
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens_value,
                temperature=temperature,
                top_p=top_p
            )
            
            execution_time = time.time() - start_time
            
            # Log response
            APILogger.log_response(client_ip, endpoint, output[:100] + "...", execution_time)
            
            logger.info("Response generated successfully")
            return {"output": output}
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            APILogger.log_error(client_ip, endpoint, str(e))
            
            logger.error(f"Error processing image URL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)