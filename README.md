---
title: BlueprintAI Inpainting Service
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# BlueprintAI Inpainting Service

This is a FastAPI service for image inpainting based on text descriptions and theme colors. The service processes images and applies AI-generated modifications according to specified themes.

## API Endpoints

### POST /inpaint/

Parameters:
- `image`: Image file to process
- `theme_description`: Description of the theme
- `theme_color`: Color hex code for the theme

Returns:
- JSON object with base64-encoded processed image