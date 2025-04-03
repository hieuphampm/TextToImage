from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware  
import torch
import torch.nn as nn
import torchvision.models as models
from diffusers import StableDiffusionPipeline
import uuid
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,                   
    allow_methods=["*"],                     
    allow_headers=["*"],                     
)

# Define the ImageEncoder class
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.feature_extrator = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        x = self.feature_extrator(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load the trained model
model_path = "best_model.pth"  # Update this path
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageEncoder().to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

@app.post("/generate")
async def generate_image(prompt: str = Form(...)):
    with torch.no_grad():
        generated_image = pipe(prompt).images[0]

    filename = f"output_{uuid.uuid4().hex}.png"
    generated_image.save(filename)

    return FileResponse(filename, media_type="image/png", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)