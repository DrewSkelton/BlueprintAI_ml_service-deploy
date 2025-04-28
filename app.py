from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import base64
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
import logging
import sys

# Set up advanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# For Hugging Face Spaces compatibility
class PredictRequest(BaseModel):
    data: list

@asynccontextmanager
async def lifespan(app):
    logger.info("Starting up ML service")
    yield
    logger.info("Shutting down ML service")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint that displays basic API information"""
    logger.info("Root endpoint accessed")
    return {
        "message": "BlueprintAI Inpainting API",
        "endpoints": {
            "POST /inpaint/": "Submit an image for inpainting with specified theme",
            "POST /api/predict": "Hugging Face Spaces compatible endpoint"
        },
        "status": "online"
    }

@app.post("/inpaint/")
async def inpaint(
    image: UploadFile = File(...),
    theme_description: str = Form(...),
    theme_color: str = Form(...)
):
    try:
        logger.info(f"Received inpaint request: {theme_description}, {theme_color}")
        logger.info(f"Image filename: {image.filename}, content_type: {image.content_type}")
        
        # Read image
        image_content = await image.read()
        logger.info(f"Image content size: {len(image_content)} bytes")
        
        if len(image_content) == 0:
            logger.error("Empty image content received")
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Add header bytes debug info
        if len(image_content) > 20:
            header_bytes = image_content[:20]
            logger.info(f"Image header bytes: {' '.join(f'{b:02x}' for b in header_bytes)}")
        
        try:
            # Try to open the image
            img = Image.open(io.BytesIO(image_content)).convert("RGB")
            logger.info(f"Image opened successfully: {img.format}, size: {img.size}")
        except UnidentifiedImageError as img_err:
            logger.error(f"Unidentified image error: {str(img_err)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(img_err)}")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # For testing, just return the same image
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    except UnidentifiedImageError as img_err:
        logger.error(f"Unidentified image error: {str(img_err)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_err)}")
    except Exception as e:
        logger.error(f"General error in inpaint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Add a simple test endpoint that doesn't require file uploads
@app.post("/test")
async def test(data: dict = Body(...)):
    """Simple test endpoint that echoes the input data"""
    logger.info(f"Test endpoint called with data: {data}")
    return {"received": data, "success": True}

# HF Spaces compatible endpoint with additional troubleshooting
@app.post("/api/predict")
async def predict(request: PredictRequest):
    try:
        logger.info(f"Received predict request, data length: {len(request.data)}")
        
        # Extract data from the Spaces format
        if len(request.data) < 3:
            logger.warning(f"Insufficient data: {len(request.data)} items")
            return {"error": "Missing required data"}
        
        # Typically HF Spaces will send base64 encoded image and parameters
        image_b64 = request.data[0]
        theme_description = request.data[1]
        theme_color = request.data[2]
        
        logger.info(f"Processing with theme: {theme_description}, color: {theme_color}")
        logger.info(f"Base64 string length: {len(image_b64)} characters")
        
        try:
            # Decode the base64 image
            image_content = base64.b64decode(image_b64)
            logger.info(f"Decoded image size: {len(image_content)} bytes")
            
            # Add header bytes debug info
            if len(image_content) > 20:
                header_bytes = image_content[:20]
                logger.info(f"Image header bytes: {' '.join(f'{b:02x}' for b in header_bytes)}")
            
            img = Image.open(io.BytesIO(image_content)).convert("RGB")
            logger.info(f"Image opened successfully: {img.format}, size: {img.size}")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return {"error": f"Invalid image data: {str(e)}"}
        
        # For testing, just return the same image
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"data": [img_str]}
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return {"error": f"Error processing request: {str(e)}"}

# Catch-all route to help debug routing issues
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path_name: str):
    logger.info(f"Catch-all route accessed: {path_name}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    
    # For GET requests, just return info
    if request.method == "GET":
        return {
            "message": "Route not found",
            "requested_path": path_name,
            "available_routes": [{"path": route.path, "methods": route.methods} for route in app.routes]
        }
    
    # For POST requests, try to read the body
    if request.method == "POST":
        try:
            body = await request.body()
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    json_body = await request.json()
                    return {
                        "message": "Route not found",
                        "requested_path": path_name,
                        "content_type": content_type,
                        "json_body": json_body
                    }
                except:
                    pass
            
            return {
                "message": "Route not found",
                "requested_path": path_name,
                "content_type": content_type,
                "body_size": len(body)
            }
        except Exception as e:
            return {
                "message": "Route not found",
                "requested_path": path_name,
                "error": str(e)
            }

# Add a specific debugging endpoint
@app.get("/debug")
async def debug(request: Request):
    client_host = request.client.host if request.client else "unknown"
    headers = dict(request.headers)
    return {
        "client": client_host,
        "headers": headers,
        "base_url": str(request.base_url),
        "app_routes": [{"path": route.path, "methods": route.methods} for route in app.routes]
    }

# Simple test image endpoint that generates a test image
@app.get("/test-image")
async def test_image():
    """Generate a test image to verify the API is working"""
    # Create a simple test image (a gradient)
    width, height = 300, 200
    img = Image.new('RGB', (width, height))
    
    # Draw a simple gradient
    for x in range(width):
        for y in range(height):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 100
            img.putpixel((x, y), (r, g, b))
    
    # Return the image as base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": img_str}

# This is not required but useful for direct testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)