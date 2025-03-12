from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import io
import base64
import uvicorn
from contextlib import asynccontextmanager

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
            "POST /inpaint/": "Submit an image for inpainting with specified theme"
        },
        "status": "online"
    }

@app.post("/inpaint/")
async def inpaint(
    image: UploadFile = File(...),
    theme_description: str = Form(...),
    theme_color: str = Form(...)
):
    # Read image
    image_content = await image.read()
    img = Image.open(io.BytesIO(image_content)).convert("RGB")
    
    # For testing, just return the same image
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {"image": img_str}

# This is not required but useful for direct testing
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)