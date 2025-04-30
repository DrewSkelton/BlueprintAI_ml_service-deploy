import requests
import base64
from PIL import Image
import io

# Test the root endpoint
response = requests.get("http://localhost:8000/")
print("Root endpoint response:", response.json())

# Test inpainting
image_path = "dock.JPG"  # Replace with your test image path
with open(image_path, "rb") as img_file:
    files = {"image": img_file}
    data = {"theme_description": "forest landscape", "theme_color": "green"}
    
    response = requests.post("http://localhost:8000/inpaint/", files=files, data=data)
    
    if response.status_code == 200:
        # Decode and save the result
        result = response.json()
        image_data = base64.b64decode(result["image"])
        result_image = Image.open(io.BytesIO(image_data))
        result_image.save("inpainted_result.jpg")
        print("Successfully saved inpainted image as 'inpainted_result.jpg'")
        print("Prompt used:", result["prompt_used"])
    else:
        print("Error:", response.status_code, response.text)