from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import io
import base64
import uvicorn
import torch
from diffusers import DiffusionPipeline
from contextlib import asynccontextmanager
import os

# Read environment variables with defaults for Docker
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/app/.cache")
MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/stable-diffusion-2-inpainting")
INFERENCE_STEPS = int(os.environ.get("INFERENCE_STEPS", "30"))
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "7.5"))
USE_LOCAL_FILES = os.environ.get("USE_LOCAL_FILES", "False").lower() == "true"

# Set up cache directories for Hugging Face
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["DIFFUSERS_CACHE"] = CACHE_DIR

# Ensure cache directory exists with proper permissions
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables for model
model = None

@asynccontextmanager
async def lifespan(app):
    print(f"Starting up ML service with model: {MODEL_ID}")
    print(f"Cache directory: {CACHE_DIR}")
    
    # Load model during startup
    global model
    try:
        model = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            local_files_only=USE_LOCAL_FILES,
        )
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("ML model loaded on GPU")
        else:
            print("ML model loaded on CPU")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    yield
    print("Shutting down ML service")
    # Clean up resources
    if model:
        del model

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    """Root endpoint that displays basic API information"""
    return {
        "message": "BlueprintAI Inpainting API",
        "endpoints": {
            "POST /inpaint/": "Submit an image for inpainting with specified theme"
        },
        "model": MODEL_ID,
        "status": "online" if model is not None else "model loading failed"
    }

@app.post("/inpaint/")
async def inpaint(
    image: UploadFile = File(...),
    theme_description: str = Form(...),
    theme_color: str = Form(...)
):
    """Process an image with an inpainting model using a hardcoded mask."""
    # Check if model is loaded
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    try:
        # Read image
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # Resize image to match model requirements (512x512 is typical for SD models)
        original_size = img.size
        img = img.resize((512, 512))
        
        # Create a hardcoded mask (white is the area to inpaint, black is the area to keep)
        mask = Image.new("RGB", (512, 512), "black")
        mask_draw = ImageDraw.Draw(mask)
        # Create a circular mask in the center
        mask_draw.ellipse(
            [(512 * 0.25, 512 * 0.25), (512 * 0.75, 512 * 0.75)],
            fill="white"
        )
        mask = mask.convert("L")  # Convert to grayscale
        
        # Prepare prompt based on theme description and color
        prompt = f"{theme_description} with {theme_color} color, high quality, detailed"
        
        # Run inference with configurable parameters
        output = model(
            prompt=prompt,
            image=img,
            mask_image=mask,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=INFERENCE_STEPS,
        ).images[0]
        
        # Resize back to original dimensions if needed
        if original_size != (512, 512):
            output = output.resize(original_size)
        
        # Convert output to base64 encoded string
        buffered = io.BytesIO()
        output.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "prompt_used": prompt,
            "mask_type": "circular center mask"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# This is not required but useful for direct testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)