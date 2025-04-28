from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import base64
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json

# For Hugging Face Spaces compatibility
class PredictRequest(BaseModel):
    data: list

@asynccontextmanager
async def lifespan(app):
    print("Starting up ML service")
    yield
    print("Shutting down ML service")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Root endpoint that displays basic API information"""
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
        # Read image
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # For testing, just return the same image
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# HF Spaces compatible endpoint
@app.post("/api/predict")
async def predict(request: PredictRequest):
    try:
        # Extract data from the Spaces format
        if len(request.data) < 3:
            raise HTTPException(status_code=400, detail="Missing required data")
        
        # Typically HF Spaces will send base64 encoded image and parameters
        image_b64 = request.data[0]
        theme_description = request.data[1]
        theme_color = request.data[2]
        
        # Decode the base64 image
        image_content = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_content)).convert("RGB")
        
        # For testing, just return the same image
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"data": [img_str]}
    except UnidentifiedImageError:
        return {"error": "Invalid image data"}
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

# This is not required but useful for direct testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)