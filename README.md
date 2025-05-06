


          
# Server README Update

I'll update the README.md to include information about installing NVIDIA CUDA drivers using the provided script. Here's the modified version:

```markdown:/Users/francescosalerno/Projects/describe-monorepo/server/README.md
# Describe Server

An API server for image analysis using Large Language Models (LLMs).

## Overview

This server provides an API endpoint for analyzing images using a multimodal LLM. It can process uploaded images and generate detailed descriptions based on the content.

## Model

This API uses the **moonshotai/Kimi-VL-A3B-Instruct** model from Hugging Face. The model is quantized in 8-bit using the BitsAndBytes library to optimize memory usage while maintaining performance. The quantization configuration includes:

- 8-bit quantization (load_in_8bit=True)
- FP32 CPU offloading for improved memory management
- Auto device mapping for optimal GPU utilization
- Float16 precision for tensor operations

For more details on the model implementation, see the `KimiVLInference` class in `/home/ubuntu/llm-api/app/kimi_vl_lib.py`.

## Features

- Image analysis and description generation
- Configurable system prompts
- Adjustable generation parameters (max tokens, temperature, top_p)
- Health check endpoint

## API Endpoints

### 1. Text Generation Endpoint

*POST /generate/text*
- Generates text responses without images
- Accepts JSON payload with:
  - `prompt` (required): The text prompt
  - `system_prompt` (optional): System instructions
  - `max_tokens` (optional): Maximum tokens to generate
  - `temperature` (optional, default 0.7): Controls randomness
  - `top_p` (optional, default 0.9): Controls diversity

### 2. Image Processing Endpoint

*POST /generate/image*
- Generates responses based on uploaded images and text prompts
- Accepts form data with:
  - `prompt` (required): The text prompt
  - `system_prompt` (optional): System instructions
  - `max_tokens` (optional): Maximum tokens to generate
  - `temperature` (optional, default 0.7): Controls randomness
  - `top_p` (optional, default 0.9): Controls diversity
  - `images` (required): One or more image files

### 3. Image URL Processing Endpoint

*POST /generate/image_url*
- Generates responses based on an image URL and text prompt
- Accepts form data with:
  - `prompt` (required): The text prompt
  - `system_prompt` (optional): System instructions
  - `max_tokens` (optional): Maximum tokens to generate
  - `temperature` (optional, default 0.7): Controls randomness
  - `top_p` (optional, default 0.9): Controls diversity
  - `image_url` (required): URL of the image to process

### 4. Health Check Endpoint
*GET /health*
- Simple health check endpoint that returns `{"status": "healthy"}`
- Used for monitoring the API's availability

All endpoints except `/health` require API key authentication via the `X-API-Key` header.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/frasal-test/describe-server.git
cd describe-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install NVIDIA CUDA drivers:
   This application requires NVIDIA CUDA drivers to be installed for GPU acceleration. A convenience script is provided to automate the installation process:
   ```bash
   ./install_cuda_drivers.sh
   ```
   The script will:
   - Update your system
   - Install NVIDIA drivers
   - Configure CUDA 12.8 repositories
   - Install CUDA Toolkit
   - Set up environment variables
   - Install PyTorch with CUDA support
   - Create and configure a Python virtual environment
   
   Note: The script will require a system reboot during the installation process.

4. Set up environment variables (see .env.example)

5. Run the server:
```bash
./start_api.sh
```
Note: this script contains a API_KEY that you can change to your own. Remember to copy it into LLM_API_KEY in the client .env file.

## Configuration
Copy .env.example to .env and adjust the settings as needed.

## License
This project is licensed under the MIT License - see the LICENSE file for details.